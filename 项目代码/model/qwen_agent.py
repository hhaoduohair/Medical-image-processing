"""
Qwen2.5-0.5B 智能体模块
解析 RAG 规范 → 生成预处理指令 / 无效样本判定结论
"""
import re
import json
import logging
from typing import Dict, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import MODEL_CONFIG

logger = logging.getLogger(__name__)


# ── Prompt 模板（核心资产） ────────────────────────────────────────────────────
PROMPT_INVALID_DETECTION = """你是医学图像预处理专家。
根据以下权威规范：
{rules}

请分析输入的【{modality}】图像的图像特征指标：
{image_features}

判断该图像是否为无效样本。请以 JSON 格式输出：
{{
  "is_invalid": true/false,
  "invalid_types": ["模糊"|"过曝"|"欠曝"|"病灶缺失"|"成像畸变"|"染色异常"|"金属伪影"],
  "confidence": 0.0~1.0,
  "reason": "判定依据（引用规范条款）"
}}
仅输出 JSON，不要额外解释。"""


PROMPT_PREPROCESSING = """你是医学图像预处理专家。
根据以下权威规范：
{rules}

针对【{modality}】图像，生成标准化预处理指令。
当前图像尺寸：{width}x{height}，通道数：{channels}

请以 JSON 格式输出预处理参数：
{{
  "steps": ["denoise", "normalize", "resize"],
  "denoise_params": {{"h": 10, "template_window": 7, "search_window": 21}},
  "target_size": [224, 224],
  "normalization": {{"method": "minmax", "range": [0, 255]}},
  "color_space": "RGB",
  "notes": "补充说明"
}}
仅输出 JSON，不要额外解释。"""


class QwenAgent:
    """Qwen2.5-0.5B 推理智能体（单例懒加载）"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def _load_model(self):
        if self._loaded:
            return
        logger.info(f"正在加载模型: {MODEL_CONFIG['model_name']} ...")
        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                     "float32": torch.float32}
        torch_dtype = dtype_map.get(MODEL_CONFIG["torch_dtype"], torch.float16)

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_CONFIG["model_name"], trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_CONFIG["model_name"],
            torch_dtype=torch_dtype,
            device_map=MODEL_CONFIG["device"],
            trust_remote_code=True,
        )
        self.model.eval()
        self._loaded = True
        logger.info("模型加载完成。")

    # ── 通用推理 ──────────────────────────────────────────────────────────────
    def _infer(self, prompt: str) -> str:
        self._load_model()
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MODEL_CONFIG["max_new_tokens"],
                temperature=MODEL_CONFIG["temperature"],
                top_p=MODEL_CONFIG["top_p"],
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return response.strip()

    @staticmethod
    def _parse_json(text: str) -> Optional[Dict]:
        """鲁棒 JSON 解析（容忍 Markdown 代码块）"""
        text = re.sub(r"```(?:json)?", "", text).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group())
                except Exception:
                    pass
        return None

    # ── 无效样本判定 ──────────────────────────────────────────────────────────
    def detect_invalid(
        self,
        modality: str,
        image_features: Dict,
        rules: str,
    ) -> Dict:
        """
        利用 LLM 判定图像是否为无效样本
        返回: {"is_invalid": bool, "invalid_types": [...], "confidence": float, "reason": str}
        """
        prompt = PROMPT_INVALID_DETECTION.format(
            rules=rules,
            modality=modality,
            image_features=json.dumps(image_features, ensure_ascii=False, indent=2),
        )
        raw = self._infer(prompt)
        result = self._parse_json(raw)
        if result is None:
            logger.warning("LLM 输出 JSON 解析失败，降级为规则判定")
            return {"is_invalid": False, "invalid_types": [], "confidence": 0.5,
                    "reason": "LLM解析失败，建议人工复核", "raw": raw}
        return result

    # ── 预处理指令生成 ────────────────────────────────────────────────────────
    def generate_preprocessing_params(
        self,
        modality: str,
        image_shape: Tuple[int, int, int],
        rules: str,
    ) -> Dict:
        """
        生成标准化预处理参数
        返回: {"steps": [...], "denoise_params": {...}, "target_size": [...], ...}
        """
        h, w = image_shape[:2]
        channels = image_shape[2] if len(image_shape) > 2 else 1
        prompt = PROMPT_PREPROCESSING.format(
            rules=rules,
            modality=modality,
            width=w,
            height=h,
            channels=channels,
        )
        raw = self._infer(prompt)
        result = self._parse_json(raw)
        if result is None:
            logger.warning("LLM 预处理参数解析失败，使用默认参数")
            return self._default_params(modality)
        return result

    @staticmethod
    def _default_params(modality: str) -> Dict:
        from config import IMAGE_CONFIG
        size = IMAGE_CONFIG["target_size"].get(modality,
               IMAGE_CONFIG["target_size"]["default"])
        denoise = IMAGE_CONFIG["denoise"].get(modality,
                  IMAGE_CONFIG["denoise"]["default"])
        return {
            "steps": ["denoise", "normalize", "resize"],
            "denoise_params": denoise,
            "target_size": list(size),
            "normalization": {"method": "minmax", "range": [0, 255]},
            "color_space": "RGB",
            "notes": "默认参数（LLM回退）",
        }