"""
预处理结果质量校验模块
"""
import logging
from dataclasses import dataclass, field
from typing import List, Tuple
import cv2
import numpy as np

from config import IMAGE_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class QualityCheckResult:
    passed: bool = True
    issues: List[str] = field(default_factory=list)
    score: float = 1.0       # 0~1，越高越好

    def to_dict(self):
        return {"passed": self.passed, "issues": self.issues, "score": self.score}


class QualityChecker:
    """预处理后图像质量校验器"""

    def check(
        self,
        image: np.ndarray,
        modality: str,
        target_size: Tuple[int, int] = None,
    ) -> QualityCheckResult:
        issues = []
        score = 1.0

        expected = target_size or IMAGE_CONFIG["target_size"].get(
            modality, IMAGE_CONFIG["target_size"]["default"]
        )

        # 1. 尺寸校验
        actual_h, actual_w = image.shape[:2]
        if (actual_h, actual_w) != tuple(expected):
            issues.append(
                f"尺寸不达标: 实际 {actual_w}x{actual_h}，期望 {expected[1]}x{expected[0]}"
            )
            score -= 0.3

        # 2. 仍然模糊
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) \
               if len(image.shape) == 3 else image
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if lap_var < IMAGE_CONFIG["invalid_thresholds"]["blur_laplacian"]:
            issues.append(f"处理后仍模糊: 拉普拉斯方差={lap_var:.1f}")
            score -= 0.2

        # 3. 像素值范围校验
        if image.min() < 0 or image.max() > 255:
            issues.append(f"像素值越界: [{image.min()}, {image.max()}]")
            score -= 0.2

        # 4. 全黑/全白校验
        if image.std() < 5:
            issues.append(f"图像过于单调: std={image.std():.2f}")
            score -= 0.3

        passed = len(issues) == 0
        score = max(0.0, score)
        return QualityCheckResult(passed=passed, issues=issues, score=round(score, 3))