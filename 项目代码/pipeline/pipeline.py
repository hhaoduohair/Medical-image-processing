"""
端到端预处理流水线
图像输入 → 模态识别 → RAG检索 → 无效样本判定 → 预处理 → 质量校验 → 输出
"""
import os
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import cv2
import numpy as np

from config import IMAGE_CONFIG, PIPELINE_CONFIG
from rag.knowledge_base import MedicalKnowledgeBase
from rag.retriever import MedicalRAGRetriever
from model.qwen_agent import QwenAgent
from processing.validator import ImageValidator, ValidationResult
from processing.preprocessor import MedicalImagePreprocessor
from processing.quality_checker import QualityChecker

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# 模态关键词映射（基于文件名/目录名启发式识别）
MODALITY_KEYWORDS = {
    "ct":          ["ct", "computed_tomography", "scan"],
    "mri":         ["mri", "magnetic", "flair", "t1", "t2"],
    "pathology":   ["pathology", "wsi", "slide", "histo", "he"],
    "dermoscopy":  ["dermoscopy", "derm", "skin", "ham", "isic"],
}


def detect_modality(filepath: str) -> str:
    """从文件路径启发式推断图像模态"""
    name_lower = Path(filepath).stem.lower() + Path(filepath).parent.name.lower()
    for modality, keywords in MODALITY_KEYWORDS.items():
        for kw in keywords:
            if kw in name_lower:
                return modality
    return "dermoscopy"   # HAM10000 默认


@dataclass_like = dict   # 简化数据容器


class PreprocessingResult:
    def __init__(self, filepath: str):
        self.filepath        = filepath
        self.modality        = "unknown"
        self.is_invalid      = False
        self.invalid_types   = []
        self.output_path     = None
        self.elapsed_secs    = 0.0
        self.quality_score   = 0.0
        self.retry_count     = 0
        self.error           = None

    def to_dict(self) -> Dict:
        return {
            "filepath":      self.filepath,
            "modality":      self.modality,
            "is_invalid":    self.is_invalid,
            "invalid_types": self.invalid_types,
            "output_path":   self.output_path,
            "elapsed_secs":  round(self.elapsed_secs, 2),
            "quality_score": self.quality_score,
            "retry_count":   self.retry_count,
            "error":         self.error,
        }


class MedicalPreprocessingPipeline:
    """端到端医学图像预处理流水线"""

    def __init__(self, enable_llm: bool = None):
        self.enable_llm   = enable_llm if enable_llm is not None \
                            else PIPELINE_CONFIG["enable_llm"]
        self.kb           = MedicalKnowledgeBase()
        self.retriever    = MedicalRAGRetriever(self.kb)
        self.validator    = ImageValidator()
        self.preprocessor = MedicalImagePreprocessor()
        self.checker      = QualityChecker()
        self.agent        = QwenAgent() if self.enable_llm else None

        logger.info(
            f"流水线初始化完成 | LLM={'启用' if self.enable_llm else '禁用'} "
            f"| 知识库规范数={self.kb.count()}"
        )

    # ── 单张图像处理 ──────────────────────────────────────────────────────────
    def process_single(
        self, filepath: str, modality: str = None
    ) -> PreprocessingResult:
        result = PreprocessingResult(filepath)
        t0 = time.time()

        try:
            # Step 1: 读取图像
            image = cv2.imread(filepath)
            if image is None:
                raise ValueError(f"无法读取图像: {filepath}")

            # Step 2: 模态识别
            result.modality = modality or detect_modality(filepath)
            logger.debug(f"[{Path(filepath).name}] 模态识别: {result.modality}")

            # Step 3: RAG 检索规范
            invalid_rules = self.retriever.get_invalid_rules(result.modality)
            preproc_rules = self.retriever.get_preprocessing_rules(result.modality)

            # Step 4: 无效样本判定
            if self.enable_llm and invalid_rules:
                val_res = self.validator.llm_assisted_validate(
                    image, result.modality, invalid_rules, self.agent
                )
            else:
                val_res = self.validator.rule_based_validate(image, result.modality)

            result.is_invalid    = val_res.is_invalid
            result.invalid_types = val_res.invalid_types

            if result.is_invalid:
                # 无效样本直接存档，不做预处理
                out_path = os.path.join(
                    PIPELINE_CONFIG["invalid_dir"], Path(filepath).name
                )
                cv2.imwrite(out_path, image)
                result.output_path = out_path
                logger.info(
                    f"[无效] {Path(filepath).name} | 类型={result.invalid_types}"
                )
            else:
                # Step 5: 获取预处理参数
                if self.enable_llm and preproc_rules:
                    params = self.agent.generate_preprocessing_params(
                        result.modality, image.shape, preproc_rules
                    )
                else:
                    params = MedicalImagePreprocessor._default_params(result.modality)

                # Step 6: 执行预处理 + 重试
                processed = image.copy()
                for attempt in range(PIPELINE_CONFIG["max_retry"] + 1):
                    processed = self.preprocessor.process(
                        processed, result.modality, params
                    )
                    qc = self.checker.check(processed, result.modality)
                    result.quality_score = qc.score
                    result.retry_count   = attempt
                    if qc.passed:
                        break
                    if attempt < PIPELINE_CONFIG["max_retry"]:
                        logger.warning(
                            f"质量校验失败(第{attempt+1}次)，重试: {qc.issues}"
                        )
                    else:
                        logger.warning(f"达到最大重试次数，强制输出: {qc.issues}")

                # Step 7: 保存输出
                out_path = os.path.join(
                    PIPELINE_CONFIG["valid_dir"], Path(filepath).name
                )
                cv2.imwrite(out_path, processed)
                result.output_path = out_path
                logger.info(
                    f"[有效] {Path(filepath).name} | "
                    f"质量={result.quality_score:.2f} | 重试={result.retry_count}"
                )

        except Exception as e:
            result.error = str(e)
            logger.error(f"处理失败 {filepath}: {e}")

        result.elapsed_secs = time.time() - t0
        return result

    # ── 批量处理 ──────────────────────────────────────────────────────────────
    def process_batch(
        self,
        input_dir: str,
        modality: str = None,
        max_samples: int = None,
    ) -> List[PreprocessingResult]:
        files = [
            str(p) for p in Path(input_dir).rglob("*")
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        if max_samples:
            files = files[:max_samples]

        logger.info(f"批量处理开始: {len(files)} 张图像 | 输入目录={input_dir}")
        results = []
        for i, fp in enumerate(files, 1):
            logger.info(f"进度 [{i}/{len(files)}] {Path(fp).name}")
            res = self.process_single(fp, modality)
            results.append(res)

        # 统计报告
        self._print_summary(results)
        if PIPELINE_CONFIG["save_report"]:
            self._save_report(results, input_dir)

        return results

    # ── 统计报告 ──────────────────────────────────────────────────────────────
    @staticmethod
    def _print_summary(results: List[PreprocessingResult]):
        total     = len(results)
        valid_n   = sum(1 for r in results if not r.is_invalid and not r.error)
        invalid_n = sum(1 for r in results if r.is_invalid)
        error_n   = sum(1 for r in results if r.error)
        avg_time  = sum(r.elapsed_secs for r in results) / total if total else 0

        print("\n" + "=" * 60)
        print("          医学图像预处理结果汇总")
        print("=" * 60)
        print(f"  总样本数       : {total}")
        print(f"  有效样本       : {valid_n} ({valid_n/total:.1%})")
        print(f"  无效样本       : {invalid_n} ({invalid_n/total:.1%})")
        print(f"  处理失败       : {error_n}")
        print(f"  单样本平均耗时 : {avg_time:.1f} 秒")
        if valid_n > 0:
            avg_q = sum(r.quality_score for r in results if not r.is_invalid) / valid_n
            print(f"  平均质量分     : {avg_q:.3f}")
        print("=" * 60 + "\n")

    @staticmethod
    def _save_report(results: List[PreprocessingResult], input_dir: str):
        report = {
            "input_dir": input_dir,
            "total": len(results),
            "results": [r.to_dict() for r in results],
        }
        report_path = os.path.join(
            PIPELINE_CONFIG["log_dir"],
            f"report_{int(time.time())}.json",
        )
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"处理报告已保存: {report_path}")