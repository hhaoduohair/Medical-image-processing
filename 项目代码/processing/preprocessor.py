"""
图像预处理算法模块（OpenCV）
降噪 / 灰度归一化 / 尺寸统一
"""
import logging
from typing import Dict, Tuple
import cv2
import numpy as np

from config import IMAGE_CONFIG

logger = logging.getLogger(__name__)


class MedicalImagePreprocessor:
    """医学图像标准化预处理器"""

    # ── 降噪 ──────────────────────────────────────────────────────────────────
    @staticmethod
    def denoise(image: np.ndarray, params: Dict) -> np.ndarray:
        h  = int(params.get("h", 10))
        tw = int(params.get("template_window", 7))
        sw = int(params.get("search_window", 21))
        if len(image.shape) == 3 and image.shape[2] == 3:
            return cv2.fastNlMeansDenoisingColored(image, None, h, h, tw, sw)
        else:
            gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return cv2.fastNlMeansDenoising(gray, None, h, tw, sw)

    # ── 尺寸统一 ─────────────────────────────────────────────────────────────
    @staticmethod
    def resize(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        h, w = image.shape[:2]
        th, tw = target_size
        if h == th and w == tw:
            return image
        # 保持纵横比 + 填充
        scale = min(th / h, tw / w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        # 中心填充
        if len(resized.shape) == 3:
            canvas = np.zeros((th, tw, resized.shape[2]), dtype=np.uint8)
        else:
            canvas = np.zeros((th, tw), dtype=np.uint8)
        pad_h = (th - new_h) // 2
        pad_w = (tw - new_w) // 2
        canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
        return canvas

    # ── 灰度归一化 ────────────────────────────────────────────────────────────
    @staticmethod
    def normalize(
        image: np.ndarray,
        method: str = "minmax",
        norm_range: Tuple[int, int] = (0, 255),
    ) -> np.ndarray:
        img_f = image.astype(np.float32)
        if method == "minmax":
            mn, mx = img_f.min(), img_f.max()
            if mx - mn < 1e-6:
                return np.zeros_like(image, dtype=np.uint8)
            img_f = (img_f - mn) / (mx - mn)
            img_f = img_f * (norm_range[1] - norm_range[0]) + norm_range[0]
        elif method == "zscore":
            mean, std = img_f.mean(), img_f.std()
            if std < 1e-6:
                std = 1.0
            img_f = (img_f - mean) / std
            # 映射到 [0, 255]
            img_f = np.clip((img_f + 3) / 6 * 255, 0, 255)
        return np.clip(img_f, 0, 255).astype(np.uint8)

    # ── Macenko H&E 染色归一化（病理专用） ────────────────────────────────────
    @staticmethod
    def stain_normalize_macenko(image: np.ndarray) -> np.ndarray:
        """简化版 Macenko 染色归一化"""
        try:
            img = image.astype(np.float32) + 1e-6
            od = -np.log(img / 255.0)
            od = od.reshape(-1, 3)
            # SVD 分解
            _, _, Vt = np.linalg.svd(od, full_matrices=False)
            plane = Vt[:2].T
            proj = od @ plane
            # 目标染色矩阵（标准 H&E）
            angle = np.arctan2(proj[:, 1], proj[:, 0])
            a_min, a_max = np.percentile(angle, 1), np.percentile(angle, 99)
            he = np.array([
                [np.cos(a_min), np.cos(a_max)],
                [np.sin(a_min), np.sin(a_max)],
            ])
            he_std = np.array([[0.65, 0.07], [0.70, 0.99], [0.29, 0.11]])
            C = np.linalg.lstsq(plane @ he, od.T, rcond=None)[0]
            normalized_od = (he_std @ C).T
            normalized = np.exp(-normalized_od) * 255
            normalized = np.clip(normalized, 0, 255).astype(np.uint8)
            return normalized.reshape(image.shape)
        except Exception as e:
            logger.warning(f"Macenko 归一化失败: {e}，跳过染色归一化")
            return image

    # ── 完整预处理流程 ────────────────────────────────────────────────────────
    def process(
        self,
        image: np.ndarray,
        modality: str,
        params: Dict = None,
    ) -> np.ndarray:
        """
        执行完整预处理
        params: 由 QwenAgent 或默认配置提供
        """
        if params is None:
            params = self._default_params(modality)

        steps = params.get("steps", ["denoise", "normalize", "resize"])

        for step in steps:
            if step == "denoise":
                image = self.denoise(
                    image, params.get("denoise_params", {"h": 10})
                )
                logger.debug(f"[{modality}] 降噪完成")

            elif step == "stain_normalize" and modality == "pathology":
                image = self.stain_normalize_macenko(image)
                logger.debug(f"[{modality}] 染色归一化完成")

            elif step == "normalize":
                norm_cfg = params.get("normalization", {"method": "minmax", "range": [0, 255]})
                image = self.normalize(
                    image,
                    method=norm_cfg.get("method", "minmax"),
                    norm_range=tuple(norm_cfg.get("range", [0, 255])),
                )
                logger.debug(f"[{modality}] 像素归一化完成")

            elif step == "resize":
                target = params.get("target_size",
                         list(IMAGE_CONFIG["target_size"].get(
                             modality, IMAGE_CONFIG["target_size"]["default"]
                         )))
                image = self.resize(image, tuple(target))
                logger.debug(f"[{modality}] 尺寸统一完成: {target}")

        return image

    @staticmethod
    def _default_params(modality: str) -> Dict:
        size = IMAGE_CONFIG["target_size"].get(
            modality, IMAGE_CONFIG["target_size"]["default"]
        )
        denoise_p = IMAGE_CONFIG["denoise"].get(
            modality, IMAGE_CONFIG["denoise"]["default"]
        )
        steps = (["denoise", "stain_normalize", "normalize", "resize"]
                 if modality == "pathology"
                 else ["denoise", "normalize", "resize"])
        return {
            "steps": steps,
            "denoise_params": denoise_p,
            "target_size": list(size),
            "normalization": {"method": "minmax", "range": [0, 255]},
        }