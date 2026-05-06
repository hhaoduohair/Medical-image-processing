"""
评测模块
对应报告中三大核心评测指标：F1值 / RAG准确率 / 端到端耗时
"""
import time
import logging
import csv
from typing import List, Dict, Tuple
from pathlib import Path

import cv2

from config import EVAL_CONFIG
from pipeline.pipeline import MedicalPreprocessingPipeline
from rag.retriever import MedicalRAGRetriever

logger = logging.getLogger(__name__)


# ── F1 计算工具 ────────────────────────────────────────────────────────────────
def compute_f1(
    y_pred: List[bool], y_true: List[bool]
) -> Tuple[float, float, float]:
    tp = sum(p and t for p, t in zip(y_pred, y_true))
    fp = sum(p and not t for p, t in zip(y_pred, y_true))
    fn = sum(not p and t for p, t in zip(y_pred, y_true))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return round(precision, 4), round(recall, 4), round(f1, 4)


class MedicalPreprocessingEvaluator:
    """三维评测器"""

    def __init__(self, pipeline: MedicalPreprocessingPipeline = None):
        self.pipeline  = pipeline or MedicalPreprocessingPipeline()
        self.retriever = self.pipeline.retriever

    # ── 指标一：无效样本筛选 F1 值 ────────────────────────────────────────────
    def eval_invalid_detection_f1(
        self,
        annotation_csv: str,
        image_dir: str,
        max_samples: int = 100,
    ) -> Dict:
        """
        annotation_csv 格式: image_id,is_invalid (0=有效, 1=无效)
        """
        # 读标注
        ground_truth: Dict[str, bool] = {}
        with open(annotation_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ground_truth[row["image_id"]] = bool(int(row["is_invalid"]))

        y_pred, y_true = [], []
        count = 0
        for img_id, gt_label in ground_truth.items():
            if count >= max_samples:
                break
            # 尝试匹配图像文件
            candidates = list(Path(image_dir).glob(f"{img_id}*"))
            if not candidates:
                continue
            result = self.pipeline.process_single(str(candidates[0]))
            y_pred.append(result.is_invalid)
            y_true.append(gt_label)
            count += 1

        precision, recall, f1 = compute_f1(y_pred, y_true)
        metrics = {
            "samples":   count,
            "precision": precision,
            "recall":    recall,
            "f1":        f1,
            "target":    0.965,
            "passed":    f1 >= 0.965,
        }
        self._log_metric("无效样本筛选 F1", f1, metrics["target"], metrics["passed"])
        return metrics

    # ── 指标二：RAG 检索准确率 ────────────────────────────────────────────────
    def eval_rag_accuracy(self, test_cases: List[Dict] = None) -> Dict:
        """
        test_cases: [{"modality": ..., "task": ..., "expected_keyword": ...}]
        若不传则使用内置测试集
        """
        if test_cases is None:
            test_cases = self._builtin_rag_test_cases()

        acc = self.retriever.accuracy_eval(test_cases)
        metrics = {
            "total":   len(test_cases),
            "accuracy": round(acc, 4),
            "target":  0.98,
            "passed":  acc >= 0.98,
        }
        self._log_metric("RAG 检索准确率", acc, metrics["target"], metrics["passed"])
        return metrics

    # ── 指标三：端到端耗时 ────────────────────────────────────────────────────
    def eval_end_to_end_latency(
        self, image_dir: str, max_samples: int = 100
    ) -> Dict:
        from pipeline.pipeline import SUPPORTED_EXTENSIONS
        files = [
            str(p) for p in Path(image_dir).rglob("*")
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
        ][:max_samples]

        if not files:
            logger.warning("未找到测试图像，跳过耗时评测")
            return {}

        # 冷启动预热（不计入统计）
        _ = self.pipeline.process_single(files[0])

        t0 = time.time()
        for fp in files:
            self.pipeline.process_single(fp)
        total_secs = time.time() - t0

        avg_secs = total_secs / len(files)
        metrics = {
            "samples":    len(files),
            "total_secs": round(total_secs, 2),
            "avg_secs":   round(avg_secs, 2),
            "target_secs": 60,
            "passed":     avg_secs <= 60,
        }
        self._log_metric("端到端单样本耗时(秒)", avg_secs, 60, metrics["passed"],
                         higher_is_better=False)
        return metrics

    # ── 综合评测报告 ──────────────────────────────────────────────────────────
    def full_eval(
        self,
        annotation_csv: str = None,
        image_dir: str = None,
        max_samples: int = 100,
    ) -> Dict:
        annotation_csv = annotation_csv or EVAL_CONFIG["annotation_file"]
        image_dir      = image_dir      or EVAL_CONFIG["dataset_path"]

        print("\n" + "=" * 60)
        print("          开始综合评测")
        print("=" * 60)

        report = {}
        report["rag_accuracy"]  = self.eval_rag_accuracy()
        report["f1_detection"]  = self.eval_invalid_detection_f1(
            annotation_csv, image_dir, max_samples
        )
        report["latency"]       = self.eval_end_to_end_latency(
            image_dir, max_samples
        )

        all_passed = all(
            v.get("passed", False) for v in report.values() if isinstance(v, dict)
        )
        print("\n" + "=" * 60)
        print(f"  综合评测结论: {'✅ 全部达标' if all_passed else '❌ 存在未达标指标'}")
        print("=" * 60 + "\n")

        return report

    # ── 内置 RAG 测试集 ───────────────────────────────────────────────────────
    @staticmethod
    def _builtin_rag_test_cases() -> List[Dict]:
        return [
            {"modality": "dermoscopy", "task": "invalid_detection",
             "expected_keyword": "拉普拉斯"},
            {"modality": "dermoscopy", "task": "preprocessing",
             "expected_keyword": "224"},
            {"modality": "pathology",  "task": "invalid_detection",
             "expected_keyword": "组织缺失"},
            {"modality": "pathology",  "task": "preprocessing",
             "expected_keyword": "Macenko"},
            {"modality": "ct",         "task": "invalid_detection",
             "expected_keyword": "金属伪影"},
            {"modality": "ct",         "task": "preprocessing",
             "expected_keyword": "512"},
            {"modality": "mri",        "task": "invalid_detection",
             "expected_keyword": "运动伪影"},
            {"modality": "mri",        "task": "preprocessing",
             "expected_keyword": "256"},
        ]

    @staticmethod
    def _log_metric(name, value, target, passed, higher_is_better=True):
        symbol = "✅" if passed else "❌"
        cmp    = ">=" if higher_is_better else "<="
        print(f"  {symbol} {name}: {value}  (目标 {cmp} {target})")