import os
import torch
from dotenv import load_dotenv
import os
load_dotenv()
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    LlamaForCausalLM,
    logging,
)
base_model="vilm/vinallama-2.7b"
from peft import LoraConfig
compute_dtype = getattr(torch, "float16")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)
print(os.getenv("token_hf"))
model = LlamaForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    device_map={"": 0},
    token=os.getenv("token_hf"),
)
model.config.use_cache = False
model.config.pretraining_tp = 1