"""
RAG 知识库构建模块
将医学预处理规范文档向量化存入 ChromaDB
"""
import os
import logging
from typing import List, Dict
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from config import RAG_CONFIG

logger = logging.getLogger(__name__)

# ── 权威规范文档（内置种子知识） ──────────────────────────────────────────────
BUILTIN_RULES: List[Dict] = [
    # ---- 皮肤病理 / 皮肤镜 ----
    {
        "id": "dermoscopy_001",
        "modality": "dermoscopy",
        "task": "invalid_detection",
        "content": (
            "皮肤镜图像无效样本判定规范（依据《数字病理图像处理技术规范》）："
            "① 模糊判定：拉普拉斯方差 < 50 判定为模糊样本；"
            "② 过曝判定：灰度值 > 250 的像素占比 > 15% 判定为过曝；"
            "③ 欠曝判定：灰度值 < 10 的像素占比 > 15% 判定为欠曝；"
            "④ 病灶缺失：有效内容像素占比 < 5% 判定为病灶缺失；"
            "⑤ 成像畸变：图像四角出现黑色圆弧遮挡面积 > 20% 判定为成像畸变。"
        ),
        "source": "《数字病理图像处理技术规范》人民卫生出版社 2024",
    },
    {
        "id": "dermoscopy_002",
        "modality": "dermoscopy",
        "task": "preprocessing",
        "content": (
            "皮肤镜图像标准化预处理规范（依据 GB/T 37358-2019）："
            "① 色彩空间：保留 RGB 三通道，不强制转灰度；"
            "② 尺寸统一：双线性插值缩放至 224×224 像素；"
            "③ 降噪：采用 fastNlMeansDenoisingColored，h=10，"
            "   templateWindowSize=7，searchWindowSize=21；"
            "④ 灰度归一化：像素值线性映射至 [0, 255]，uint8 存储；"
            "⑤ 操作顺序：降噪 → 尺寸统一 → 归一化。"
        ),
        "source": "GB/T 37358-2019 医疗影像处理与分析操作规范",
    },
    # ---- 病理切片 ----
    {
        "id": "pathology_001",
        "modality": "pathology",
        "task": "invalid_detection",
        "content": (
            "病理切片图像无效样本判定规范："
            "① 模糊判定：拉普拉斯方差 < 50；"
            "② 过曝/欠曝：同皮肤镜标准；"
            "③ 组织缺失：前景像素（非背景白色）占比 < 10% 判定为组织缺失；"
            "④ 染色异常：H&E 染色图像 R 通道均值 < 80 或 B 通道均值 < 80 "
            "   且两者差异 > 100 时判定为染色异常。"
        ),
        "source": "《数字病理图像处理技术规范》人民卫生出版社 2024",
    },
    {
        "id": "pathology_002",
        "modality": "pathology",
        "task": "preprocessing",
        "content": (
            "病理切片图像标准化预处理规范："
            "① 尺寸统一：双线性插值缩放至 224×224；"
            "② 降噪：fastNlMeansDenoisingColored，h=10；"
            "③ 色彩归一化：使用 Macenko 方法进行 H&E 染色归一化；"
            "④ 操作顺序：降噪 → 染色归一化 → 尺寸统一 → 像素归一化。"
        ),
        "source": "《数字病理图像处理技术规范》人民卫生出版社 2024",
    },
    # ---- CT ----
    {
        "id": "ct_001",
        "modality": "ct",
        "task": "invalid_detection",
        "content": (
            "CT 图像无效样本判定规范："
            "① 模糊判定：拉普拉斯方差 < 50；"
            "② 金属伪影：高亮条纹像素（灰度 > 240）呈辐射状分布占比 > 5%；"
            "③ 截断伪影：图像边缘 10px 内存在连续亮带；"
            "④ 成像不全：前景像素占比 < 5%。"
        ),
        "source": "GB/T 37358-2019 医疗影像处理与分析操作规范",
    },
    {
        "id": "ct_002",
        "modality": "ct",
        "task": "preprocessing",
        "content": (
            "CT 图像标准化预处理规范："
            "① 窗宽窗位：软组织窗 WL=40 HU，WW=400 HU；"
            "② 尺寸统一：双线性插值缩放至 512×512；"
            "③ 降噪：fastNlMeansDenoising（灰度），h=8；"
            "④ 归一化：将 HU 值裁剪至 [-160, 240] 后线性映射至 [0, 255]；"
            "⑤ 操作顺序：窗宽窗位 → 降噪 → 尺寸统一 → 归一化。"
        ),
        "source": "GB/T 37358-2019 医疗影像处理与分析操作规范",
    },
    # ---- MRI ----
    {
        "id": "mri_001",
        "modality": "mri",
        "task": "invalid_detection",
        "content": (
            "MRI 图像无效样本判定规范："
            "① 模糊判定：拉普拉斯方差 < 50；"
            "② 运动伪影：图像边缘存在重影，Sobel 梯度能量异常高 > 阈值 1.5x；"
            "③ 磁场不均匀：图像四角亮度差异 > 30%；"
            "④ 成像不全：前景像素占比 < 5%。"
        ),
        "source": "GB/T 37358-2019 医疗影像处理与分析操作规范",
    },
    {
        "id": "mri_002",
        "modality": "mri",
        "task": "preprocessing",
        "content": (
            "MRI 图像标准化预处理规范："
            "① 偏置场校正：N4ITK 算法去除磁场不均匀性；"
            "② 尺寸统一：双线性插值缩放至 256×256；"
            "③ 降噪：fastNlMeansDenoising，h=6；"
            "④ 归一化：Z-score 归一化后映射至 [0, 255]；"
            "⑤ 操作顺序：偏置场校正 → 降噪 → 尺寸统一 → 归一化。"
        ),
        "source": "GB/T 37358-2019 医疗影像处理与分析操作规范",
    },
]


class MedicalKnowledgeBase:
    """医学图像预处理规范知识库（ChromaDB）"""

    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=RAG_CONFIG["chroma_persist_dir"],
            settings=Settings(anonymized_telemetry=False),
        )
        self.embed_model = SentenceTransformer(RAG_CONFIG["embedding_model"])
        self.collection = self.client.get_or_create_collection(
            name=RAG_CONFIG["collection_name"],
            metadata={"hnsw:space": "cosine"},
        )
        self._init_builtin_rules()

    # ── 内置规范初始化 ────────────────────────────────────────────────────────
    def _init_builtin_rules(self):
        """若知识库为空则写入内置规范"""
        existing = self.collection.count()
        if existing == 0:
            logger.info("知识库为空，正在写入内置医学规范 ...")
            self.add_documents(BUILTIN_RULES)
            logger.info(f"内置规范写入完成，共 {len(BUILTIN_RULES)} 条。")
        else:
            logger.info(f"知识库已有 {existing} 条规范，跳过初始化。")

    # ── 外部文档写入 ──────────────────────────────────────────────────────────
    def add_documents(self, docs: List[Dict]):
        """
        批量写入文档
        docs: [{"id": ..., "modality": ..., "task": ..., "content": ..., "source": ...}]
        """
        ids, embeddings, documents, metadatas = [], [], [], []
        for doc in docs:
            ids.append(doc["id"])
            documents.append(doc["content"])
            embeddings.append(
                self.embed_model.encode(doc["content"]).tolist()
            )
            metadatas.append({
                "modality": doc.get("modality", "unknown"),
                "task":     doc.get("task",     "unknown"),
                "source":   doc.get("source",   ""),
            })
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def add_from_file(self, filepath: str, modality: str, task: str, source: str = ""):
        """从文本文件按段落分割写入知识库"""
        with open(filepath, "r", encoding="utf-8") as f:
            raw = f.read()
        paragraphs = [p.strip() for p in raw.split("\n\n") if len(p.strip()) > 20]
        docs = [
            {
                "id":       f"{os.path.basename(filepath)}_{i}",
                "modality": modality,
                "task":     task,
                "content":  para,
                "source":   source or filepath,
            }
            for i, para in enumerate(paragraphs)
        ]
        self.add_documents(docs)
        logger.info(f"从 {filepath} 写入 {len(docs)} 条规范。")

    def count(self) -> int:
        return self.collection.count()