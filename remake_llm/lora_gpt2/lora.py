import os
from peft import LoraConfig, inject_adapter_in_model, AutoPeftModel, PeftConfig
from peft import get_peft_model, TaskType
from transformers import AutoTokenizer, AutoModel

ckp = "jinaai/jina-reranker-v1-tiny-en"
ppm = AutoPeftModel.from_pretrained(ckp)
model = AutoModel.from_pretrained(ckp)

peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
peft_model = get_peft_model(model, peft_config)
peft_model.print_trainable_parameters()
ppm.print_trainable_parameters()
