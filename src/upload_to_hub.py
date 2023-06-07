import os
import sys
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset
from transformers import GenerationConfig
from peft import PeftModel
"""
Unused imports:
import torch.nn as nn
"""
import bitsandbytes as bnb

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)

from modeling_mpt import MPTForCausalLM
from adapt_tokenizer import AutoTokenizerForMOD

from utils.prompter import Prompter

device_map = {"": 0}

model = MPTForCausalLM.from_pretrained(
    'mosaicml/mpt-7b-instruct',
    #config=config,
    trust_remote_code=True,
    # base_model,
    #load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map=device_map,
    # quantization_config=quantization_config,
    # load_in_8bit_fp32_cpu_offload=True
)
model.save_pretrained("./SavedModels/MPT-7B-Instruct")
#model.push_to_hub("MPT-7b-Instruct-LoRA")

