"""
无效样本检测模块（基于 OpenCV 规则 + LLM 协同）
"""
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import cv2
import numpy as np

from config import IMAGE_CONFIG

logger = logging.getLogger(__name__)

INVALID_TYPES = {
    "blur":          "模糊",
    "overexposure":  "过曝",
    "underexposure": "欠曝",
    "missing_roi":   "病灶缺失",
    "distortion":    "成像畸变",
    "stain_anomaly": "染色异常",
}


@dataclass
class ValidationResult:
    is_invalid: bool = False
    invalid_types: List[str] = field(default_factory=list)
    confidence: float = 1.0
    reason: str = ""
    features: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "is_invalid":    self.is_invalid,
            "invalid_types": self.invalid_types,
            "confidence":    self.confidence,
            "reason":        self.reason,
            "features":      self.features,
        }


class ImageValidator:
    """基于 OpenCV 规则的无效样本检测器"""

    def __init__(self):
        self.thresh = IMAGE_CONFIG["invalid_thresholds"]

    # ── 特征提取 ──────────────────────────────────────────────────────────────
    def extract_features(self, image: np.ndarray) -> Dict:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) \
               if len(image.shape) == 3 else image

        total_px = gray.size
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        overexp_ratio = float(np.sum(gray > self.thresh["overexposure_pixel"]) / total_px)
        underexp_ratio = float(np.sum(gray < self.thresh["underexposure_pixel"]) / total_px)

        # 有效内容比例（非极值像素）
        content_mask = (gray > self.thresh["underexposure_pixel"]) & \
                       (gray < self.thresh["overexposure_pixel"])
        content_ratio = float(np.sum(content_mask) / total_px)

        # 边缘能量（检测畸变/成像不全）
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = float(np.sum(edges > 0) / total_px)

        features = {
            "laplacian_var":    round(laplacian_var, 4),
            "overexp_ratio":    round(overexp_ratio, 4),
            "underexp_ratio":   round(underexp_ratio, 4),
            "content_ratio":    round(content_ratio, 4),
            "edge_ratio":       round(edge_ratio, 4),
            "mean_brightness":  round(float(gray.mean()), 2),
            "std_brightness":   round(float(gray.std()), 2),
            "shape":            list(image.shape),
        }

        # 彩色图像额外特征
        if len(image.shape) == 3:
            features["channel_means"] = [
                round(float(image[:, :, c].mean()), 2) for c in range(3)
            ]

        return features

    # ── 规则检测 ──────────────────────────────────────────────────────────────
    def rule_based_validate(
        self, image: np.ndarray, modality: str = "default"
    ) -> ValidationResult:
        features = self.extract_features(image)
        result = ValidationResult(features=features)
        reasons = []

        # 模糊
        if features["laplacian_var"] < self.thresh["blur_laplacian"]:
            result.invalid_types.append(INVALID_TYPES["blur"])
            reasons.append(
                f"拉普拉斯方差={features['laplacian_var']:.1f} < {self.thresh['blur_laplacian']}"
            )

        # 过曝
        if features["overexp_ratio"] > self.thresh["overexposure_ratio"]:
            result.invalid_types.append(INVALID_TYPES["overexposure"])
            reasons.append(
                f"过曝像素占比={features['overexp_ratio']:.2%} > {self.thresh['overexposure_ratio']:.0%}"
            )

        # 欠曝
        if features["underexp_ratio"] > self.thresh["underexposure_ratio"]:
            result.invalid_types.append(INVALID_TYPES["underexposure"])
            reasons.append(
                f"欠曝像素占比={features['underexp_ratio']:.2%} > {self.thresh['underexposure_ratio']:.0%}"
            )

        # 病灶缺失
        if features["content_ratio"] < self.thresh["min_content_ratio"]:
            result.invalid_types.append(INVALID_TYPES["missing_roi"])
            reasons.append(
                f"有效内容占比={features['content_ratio']:.2%} < {self.thresh['min_content_ratio']:.0%}"
            )

        # 染色异常（病理/皮肤镜）
        if modality in ("pathology", "dermoscopy") and "channel_means" in features:
            r_mean, g_mean, b_mean = features["channel_means"]
            if (r_mean < 80 or b_mean < 80) and abs(r_mean - b_mean) > 100:
                result.invalid_types.append(INVALID_TYPES["stain_anomaly"])
                reasons.append(
                    f"染色异常: R={r_mean:.0f}, B={b_mean:.0f}, 差值={abs(r_mean-b_mean):.0f}"
                )

        result.is_invalid = len(result.invalid_types) > 0
        result.reason = "；".join(reasons) if reasons else "符合有效样本规范"
        result.confidence = 0.95 if result.is_invalid else 0.92
        return result

    # ── LLM 辅助检测（融合规则特征） ─────────────────────────────────────────
    def llm_assisted_validate(
        self,
        image: np.ndarray,
        modality: str,
        rules: str,
        agent,          # QwenAgent 实例
    ) -> ValidationResult:
        features = self.extract_features(image)
        rule_result = self.rule_based_validate(image, modality)

        llm_result = agent.detect_invalid(modality, features, rules)

        # 融合：任一判为无效则判为无效，取最高置信度
        is_invalid = rule_result.is_invalid or llm_result.get("is_invalid", False)
        invalid_types = list(set(
            rule_result.invalid_types + llm_result.get("invalid_types", [])
        ))
        confidence = max(rule_result.confidence, llm_result.get("confidence", 0.5))
        reason = (
            f"[规则] {rule_result.reason} | "
            f"[LLM] {llm_result.get('reason', '')}"
        )

        return ValidationResult(
            is_invalid=is_invalid,
            invalid_types=invalid_types,
            confidence=confidence,
            reason=reason,
            features=features,
        )