import lora_chain
from modeling_mpt import MPTForCausalLM
from adapt_tokenizer import AutoTokenizerForMOD
import torch 
import gc 
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
)

base_model = 'mosaicml/mpt-7b-instruct'
#Most recent checkpoint 
lora_weights = 'lora-mpt1'
#name of directory that the new weights/LoRA checkpoint will be saved into. The value lora-mpt is already taken (as is lora-mpt1) so it will be automatically adjusted to lora-mpt2
new_model_id = "lora-mpt"


#lora params
lora_r = 4
lora_alpha = 16
lora_target_modules = ['Wqkv']
lora_dropout = 0.05

use_gradient_checkpointing=True

model, checkpoint_chain = lora_chain.load(base_model, lora_weights)

model = prepare_model_for_int8_training(model, use_gradient_checkpointing=use_gradient_checkpointing)
config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    target_modules=lora_target_modules,
    lora_dropout=lora_dropout,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, config)

# This is where you would do the finetuning
# finetune(model, data, etc)

# Save after finetuning
lora_chain.save(model, checkpoint_chain, new_model_id)