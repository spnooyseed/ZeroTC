import os
import logging
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModel
)
from .config import models
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, embeddings1, embeddings2, labels):
        # 计算嵌入之间的欧氏距离
        print(embeddings1.shape , embeddings2.shape , labels)
        euclidean_distance = F.pairwise_distance(embeddings1, embeddings2)
        # 计算对比损失
        print('euclidean_distance = ' , euclidean_distance.shape)
        loss_contrastive = torch.mean((1 - labels) * torch.pow(euclidean_distance, 2) +
                                     (labels) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        print(f'loss_contrastive = {euclidean_distance}')
        return loss_contrastive


class ZeroTC(nn.Module) :
    def __init__(self , model_name , num_class=4 , hidden_size=768) :
        super(ZeroTC , self).__init__()
        # import pdb
        # pdb.set_trace()
        self.encoder = load_model(model_name)
        self.tokenizer = load_tokenizer(model_name)
        self.fc = nn.Linear(hidden_size , num_class)
        self.max_text_length = 128

    def contrastive_loss(self, embed1, embed2, target, mask=None, clip_prob=None):
        logits = self.sim(embed1, embed2)
        if mask is not None:
            logits = logits.masked_fill(mask, float('-inf'))
        log_prob = F.log_softmax(logits, dim=-1)
        if mask is not None:
            return -log_prob[target.bool()].mean() 
        mean_log_prob_pos = (target * log_prob).sum(1) / (target.sum(1)+1e-10)
        return -mean_log_prob_pos.mean()

    def forward(self , text1) :
        # import pdb
        # pdb.set_trace()
        text1_inputs = self.tokenizer(text1, return_tensors="pt", padding=True, truncation=True , max_length=self.max_text_length)
        # text2_inputs = self.tokenizer(text2, return_tensors="pt", padding=True, truncation=True, return_special_tokens_mask=True)
        text1_embedding = self.encoder(**text1_inputs).pooler_output
        # text2_embedding = self.encoder(**text2_inputs).last_hidden_state
        return text1_embedding


# only read offline model due to interest
def load_model(model_name) :

    save_dir = os.path.join(models , f'{model_name}/{model_name}.pt')

    if os.path.exists(save_dir) :
        logging.info(f'---- load model from cache {save_dir} ----')
        return torch.load(save_dir)

    logging.info(' create new Model {model_name} , and save to {save_dir}')
    model = AutoModel.from_pretrained(f'{models}/{model_name}')
    torch.save(model , save_dir)
    return model

def load_tokenizer(model_name) :
    path = f'{models}/{model_name}'
    return AutoTokenizer.from_pretrained(path , cache_dir=path)