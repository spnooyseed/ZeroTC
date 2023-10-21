import os
import logging
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModel
)
from .config import models

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


def train(simcse_encoder , simcse_tokenizer , train_data) :

    # 定义损失函数和优化器
    contrastive_loss = ContrastiveLoss()
    optimizer = torch.optim.Adam(simcse_encoder.parameters(), lr=0.001)

    # 进行自训练循环
    simcse_encoder.train()
    epochs = 10

    for epoch in range(epochs):
        for data , label , label_name  in tqdm(train_data):
            print(data)
            print(label)
            print(label_name)
            inputs1 = simcse_tokenizer(data, return_tensors="pt", padding=True, truncation=True)
            embeddings1 = simcse_encoder(**inputs1).last_hidden_state
            inputs2 = simcse_tokenizer(label_name, return_tensors="pt", padding=True, truncation=True)
            embeddings2 = simcse_encoder(**inputs2).last_hidden_state
            embeddings1 = embeddings1.mean(dim = 1)
            embeddings2 = embeddings2.mean(dim = 1)
            # 根据嵌入计算对比损失并进行反向传播
            # import pdb
            # pdb.set_trace()
            print('test')
            loss = contrastive_loss(embeddings1, embeddings2, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
