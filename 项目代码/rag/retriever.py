"""
RAG 检索模块
根据图像模态 + 预处理任务，从知识库检索最相关的权威规范
"""
import logging
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer

from config import RAG_CONFIG
from rag.knowledge_base import MedicalKnowledgeBase

logger = logging.getLogger(__name__)


class MedicalRAGRetriever:
    """医学规范 RAG 检索器"""

    def __init__(self, knowledge_base: Optional[MedicalKnowledgeBase] = None):
        self.kb = knowledge_base or MedicalKnowledgeBase()
        self.embed_model = SentenceTransformer(RAG_CONFIG["embedding_model"])
        self._cache: Dict[str, List[Dict]] = {}

    # ── 核心检索 ──────────────────────────────────────────────────────────────
    def retrieve(
        self,
        modality: str,
        task: str,
        query: Optional[str] = None,
        top_k: int = None,
    ) -> List[Dict]:
        """
        检索与 modality+task 最匹配的规范
        返回: [{"content": ..., "source": ..., "score": ...}]
        """
        top_k = top_k or RAG_CONFIG["top_k"]
        cache_key = f"{modality}_{task}"

        # 命中缓存直接返回
        if query is None and cache_key in self._cache:
            return self._cache[cache_key]

        # 构造查询文本
        query_text = query or f"{modality}图像{task}规范"
        query_vec = self.embed_model.encode(query_text).tolist()

        # ChromaDB 检索（优先按模态过滤）
        where_filter = {"$and": [{"modality": modality}, {"task": task}]}
        results = self.kb.collection.query(
            query_embeddings=[query_vec],
            n_results=min(top_k, self.kb.count()),
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        # 若模态过滤无结果，退回全库检索
        if not results["documents"][0]:
            logger.warning(f"模态过滤无结果，退回全库检索: {modality}/{task}")
            results = self.kb.collection.query(
                query_embeddings=[query_vec],
                n_results=min(top_k, self.kb.count()),
                include=["documents", "metadatas", "distances"],
            )

        docs = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            score = 1.0 - dist  # cosine distance → similarity
            if score >= RAG_CONFIG["score_threshold"]:
                docs.append({
                    "content":  doc,
                    "source":   meta.get("source", ""),
                    "modality": meta.get("modality", ""),
                    "task":     meta.get("task", ""),
                    "score":    round(score, 4),
                })

        if query is None:
            self._cache[cache_key] = docs

        logger.info(
            f"[RAG] modality={modality} task={task} → 检索到 {len(docs)} 条规范"
        )
        return docs

    # ── 便捷方法 ──────────────────────────────────────────────────────────────
    def get_invalid_rules(self, modality: str) -> str:
        """返回拼接后的无效样本判定规范文本"""
        docs = self.retrieve(modality, "invalid_detection")
        return "\n\n".join(d["content"] for d in docs) if docs else ""

    def get_preprocessing_rules(self, modality: str) -> str:
        """返回拼接后的预处理规范文本"""
        docs = self.retrieve(modality, "preprocessing")
        return "\n\n".join(d["content"] for d in docs) if docs else ""

    def accuracy_eval(self, test_cases: List[Dict]) -> float:
        """
        评测检索准确率
        test_cases: [{"modality": ..., "task": ..., "expected_keyword": ...}]
        """
        correct = 0
        for case in test_cases:
            docs = self.retrieve(case["modality"], case["task"])
            combined = " ".join(d["content"] for d in docs)
            if case["expected_keyword"] in combined:
                correct += 1
        acc = correct / len(test_cases) if test_cases else 0.0
        logger.info(f"RAG 检索准确率: {acc:.2%} ({correct}/{len(test_cases)})")
        return acc