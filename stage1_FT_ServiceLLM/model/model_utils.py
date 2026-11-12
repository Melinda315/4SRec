import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_base_and_tokenizer(base_model_name, use_4bit, bf16):
    cuda = torch.cuda.is_available()
    if use_4bit and not cuda:
        raise RuntimeError("4-bit mode requires CUDA")
    if bf16 and (not cuda or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("bf16 mode requires a CUDA GPU with bf16 support")
    dtype = torch.bfloat16 if bf16 else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    quantization = None
    if use_4bit:
        quantization = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
        )
    options = {"quantization_config": quantization} if quantization else {"torch_dtype": dtype}
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, device_map="auto" if cuda else None, **options,
    )
    return base_model, tokenizer


def load_model_and_tokenizer(
    base_model_name, lora_r, lora_alpha, lora_dropout, lora_target_modules, use_4bit, bf16,
):
    base_model, tokenizer = load_base_and_tokenizer(base_model_name, use_4bit, bf16)
    if use_4bit:
        base_model = prepare_model_for_kbit_training(base_model)
    lora = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias="none",
        task_type="CAUSAL_LM", target_modules=lora_target_modules,
    )
    return get_peft_model(base_model, lora), tokenizer
