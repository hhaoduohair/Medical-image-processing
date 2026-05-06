"""
全局配置文件
基于大模型的医学图像自动化预处理工具 - 第四组
"""
import os

MODEL_CONFIG = {
    "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
    "max_new_tokens": 512,
    "temperature": 0.1,
    "top_p": 0.9,
    "device": "auto",          # "cpu" / "cuda" / "auto"
    "torch_dtype": "float16",
}

RAG_CONFIG = {
    "chroma_persist_dir": "./chroma_db",
    "collection_name": "medical_preprocessing_rules",
    "embedding_model": "BAAI/bge-small-zh-v1.5",  # 轻量化中文嵌入模型
    "top_k": 3,
    "score_threshold": 0.5,
}

IMAGE_CONFIG = {
    "target_size": {
        "pathology":  (224, 224),
        "dermoscopy": (224, 224),
        "ct":         (512, 512),
        "mri":        (256, 256),
        "default":    (224, 224),
    },
    "denoise": {
        "pathology":  {"h": 10, "template_window": 7, "search_window": 21},
        "dermoscopy": {"h": 10, "template_window": 7, "search_window": 21},
        "ct":         {"h": 8,  "template_window": 7, "search_window": 21},
        "mri":        {"h": 6,  "template_window": 7, "search_window": 21},
        "default":    {"h": 10, "template_window": 7, "search_window": 21},
    },
    "invalid_thresholds": {
        "blur_laplacian":      50.0,
        "overexposure_ratio":  0.15,
        "underexposure_ratio": 0.15,
        "overexposure_pixel":  250,
        "underexposure_pixel": 10,
        "min_content_ratio":   0.05,
    },
}

PIPELINE_CONFIG = {
    "output_dir":  "./output",
    "invalid_dir": "./output/invalid",
    "valid_dir":   "./output/valid",
    "log_dir":     "./logs",
    "batch_size":  10,
    "max_retry":   2,
    "enable_llm":  True,
    "save_report": True,
}

EVAL_CONFIG = {
    "dataset_path":    "./data/HAM10000",
    "annotation_file": "./data/HAM10000/labels.csv",
    "eval_output":     "./eval_results",
}

for _d in [
    PIPELINE_CONFIG["output_dir"], PIPELINE_CONFIG["invalid_dir"],
    PIPELINE_CONFIG["valid_dir"],  PIPELINE_CONFIG["log_dir"],
    RAG_CONFIG["chroma_persist_dir"], EVAL_CONFIG["eval_output"],
]:
    os.makedirs(_d, exist_ok=True)