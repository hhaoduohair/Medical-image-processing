"""
程序入口
用法：
  python main.py --mode batch --input ./data/HAM10000
  python main.py --mode single --input ./data/example.jpg
  python main.py --mode eval
"""
import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("./logs/run.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="基于大模型的医学图像自动化预处理工具"
    )
    parser.add_argument(
        "--mode", choices=["single", "batch", "eval", "build_kb"],
        default="batch", help="运行模式"
    )
    parser.add_argument("--input",    default="./data", help="输入图像路径或目录")
    parser.add_argument("--modality", default=None,
                        help="手动指定模态(ct/mri/pathology/dermoscopy)")
    parser.add_argument("--max",      type=int, default=100, help="最大处理样本数")
    parser.add_argument("--no-llm",   action="store_true", help="禁用 LLM，仅规则模式")
    parser.add_argument("--ann",      default=None, help="标注 CSV（eval 模式）")
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info(f"启动模式: {args.mode}")

    # ── 构建/更新知识库 ─────────────────────────────────────────────────────
    if args.mode == "build_kb":
        from rag.knowledge_base import MedicalKnowledgeBase
        kb = MedicalKnowledgeBase()
        logger.info(f"知识库共 {kb.count()} 条规范（内置规范已自动加载）")
        print("如需添加自定义规范文档，请调用 kb.add_from_file(filepath, modality, task)")
        return

    # ── 单张处理 ────────────────────────────────────────────────────────────
    if args.mode == "single":
        from pipeline.pipeline import MedicalPreprocessingPipeline
        pipeline = MedicalPreprocessingPipeline(enable_llm=not args.no_llm)
        result = pipeline.process_single(args.input, args.modality)
        print("\n处理结果:")
        import json
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return

    # ── 批量处理 ────────────────────────────────────────────────────────────
    if args.mode == "batch":
        from pipeline.pipeline import MedicalPreprocessingPipeline
        pipeline = MedicalPreprocessingPipeline(enable_llm=not args.no_llm)
        results = pipeline.process_batch(args.input, args.modality, args.max)
        logger.info(f"批量处理完成，共处理 {len(results)} 张图像。")
        return

    # ── 综合评测 ────────────────────────────────────────────────────────────
    if args.mode == "eval":
        from pipeline.pipeline import MedicalPreprocessingPipeline
        from eval.evaluator import MedicalPreprocessingEvaluator
        pipeline  = MedicalPreprocessingPipeline(enable_llm=not args.no_llm)
        evaluator = MedicalPreprocessingEvaluator(pipeline)
        report    = evaluator.full_eval(
            annotation_csv=args.ann,
            image_dir=args.input,
            max_samples=args.max,
        )
        import json
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()