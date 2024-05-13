from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
import argparse
from pathlib import Path
from glob import glob
import os 
 
name="LanguageBind/Video-LLaVA-7B"
name="/data/home/shilongwang/workplace/Multimodal-LLaVA/checkpoints/Video-LLaVA-7B"

tokenizer = LlamaTokenizer.from_pretrained(name)
tokenizer.save_pretrained('checkpoints/Video-LLaVA-7B-split')

model = LlamaForCausalLM.from_pretrained(name, torch_dtype=torch.float16)
model.save_pretrained('checkpoints/Video-LLaVA-7B-split')

mm_projector_weight={}
for file_name in [
    'pytorch_model-00001-of-00002.bin',
    'pytorch_model-00002-of-00002.bin',
]:
    main_llm_weights={}
    temp=torch.load(f'checkpoints/Video-LLaVA-7B-split/{file_name}')
    for one_name in temp:
        if 'mm_projector' in one_name:
            mm_projector_weight[one_name]=temp[one_name]
        else:
            main_llm_weights[one_name]=temp[one_name]
    torch.save(main_llm_weights,f'checkpoints/Video-LLaVA-7B-split/{file_name}')
torch.save(mm_projector_weight,'checkpoints/Video-LLaVA-7B-split/mm_projector.bin')

os.system('cp checkpoints/Video-LLaVA-7B-split/pytorch_model.bin.index.backup.json checkpoints/Video-LLaVA-7B-split/pytorch_model.bin.index.json')



temp=torch.load('/data/home/shilongwang/workplace/A_pretrain_models/Video-LLaVA-7B-split/mm_projector.bin')
print(temp)


