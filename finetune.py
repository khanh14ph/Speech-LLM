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
base_model="Viet-Mistral/Vistral-7B-Chat"
guanaco_dataset = "mlabonne/guanaco-llama2-1k"
new_model = "llama-2-7b-chat-guanaco"

dataset = load_dataset(
    "csv",
    data_files="/home4/khanhnd/self-condition/data/valset_bilingual.tsv",
    split="train",
    cache_dir="cache",
    sep="\t",
)
print(dataset)
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


tokenizer = AutoTokenizer.from_pretrained(
    base_model,
    padding_side="left",
    add_eos_token=True,
    add_bos_token=True, 
)
tokenizer.pad_token = tokenizer.eos_token
max_length=100
def tokenize(prompt):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        padding="longest",
    )
    result["labels"] = result["input_ids"].copy()
    print(result)
    return result

def generate_and_tokenize_prompt(data_point):

    return tokenize(data_point["transcript"])
dataset = dataset.map(generate_and_tokenize_prompt)
untokenized_text = tokenizer.decode(dataset[1]['input_ids']) 
print(untokenized_text)
eval_tokenizer = AutoTokenizer.from_pretrained(
    base_model,
    add_bos_token=True,
)

conversation = [{"role": "system", "content": "Bạn là một trợ lí AI hữu ích. Hãy trả lời người dùng một cách chính xác." }]

conversation.append({"role": "user", "content": "Trong một đoạn hội thoại giữa người và trợ lí ảo. Câu trước của câu:'cho tôi đăng kí gói cước b50' có thể là gì ?" })
input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt").to(model.device)


model.eval()
with torch.no_grad():
    print(eval_tokenizer.batch_decode(model.generate(
        input_ids=input_ids,
        max_new_tokens=768,
        do_sample=True,
        top_p=0.95,
        top_k=40,
        temperature=0.1,
        repetition_penalty=1.05,
    )[:, input_ids.size(1): ], skip_special_tokens=True)[0].strip())


#Set Up LoRA
from peft import prepare_model_for_kbit_training

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )