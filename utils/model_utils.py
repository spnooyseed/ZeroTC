import os
import logging
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModel
)
from .config import models

def load_model(model_name) :

    save_dir = os.path.join(models , f'{model_name}/{model_name}.pt')

    if os.path.exists(save_dir) :
        logging.info(f'---- load model from cache {save_dir} ----')
        return torch.load(save_dir)

    logging.info(' create new Model {model_name} , and save to {save_dir}')
    model = AutoModel.from_pretrained(models + f'/{model_name}' , local_files_only= True)
    torch.save(model , save_dir)
    return model

def load_tokenizer(model_name) :
    path = f'{models}/{model_name}'
    return AutoTokenizer.from_pretrained(path  , cache_dir=path)