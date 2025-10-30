from __future__ import annotations
import os, torch
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import inspect
import warnings

# Quiet only this specific advisory from transformers
warnings.filterwarnings(
    "ignore",
    message="`torch_dtype` is deprecated! Use `dtype` instead!"
)
# silence tokenizers parallelism warning globally
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def pick_device(pref: str = "auto") -> str:
    if pref == "cpu": return "cpu"
    if pref == "cuda": return "cuda" if torch.cuda.is_available() else "cpu"
    if pref == "mps":
        mps = getattr(torch.backends, "mps", None)
        return "mps" if (mps and mps.is_available()) else "cpu"
    if torch.cuda.is_available(): return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps and mps.is_available(): return "mps"
    return "cpu"

class LocalHFGenerator:
    """
    Lightweight local generator using a small instruct model.
    Default: TinyLlama/TinyLlama-1.1B-Chat-v1.0 (fits on laptop; OK on MPS).
    """
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                 device_pref: str = "auto"):
        self.model_name = model_name
        self.device = pick_device(device_pref)

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Choose dtype: fp16 for CUDA/MPS, fp32 for CPU
        dtype_val = torch.float16 if self.device in ("cuda", "mps") else torch.float32

        # Prefer new `dtype` kwarg; fall back to legacy `torch_dtype`
        kwargs = {}
        if "dtype" in inspect.signature(AutoModelForCausalLM.from_pretrained).parameters:
            kwargs["dtype"] = dtype_val
        else:
            kwargs["torch_dtype"] = dtype_val  # legacy fallback for older Transformers

        # Load and move model
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.model.to(self.device)
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 128, temperature: float = 0.2) -> str:
        toks = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **toks,
                do_sample=(temperature > 0.0),
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        # Return only the model's completion after 'Answer:'
        split = text.split("Answer:", 1)
        return split[-1].strip() if len(split) > 1 else text.strip()