import numpy as np
from .model_utils import ZeroTC , ContrastiveLoss
import logging
from torch.utils.data import DataLoader
import torch
import tqdm
from .data_utils import PseudoLabelDataset
import torch.nn.functional as F
from tqdm import tqdm

def get_accuracy(predict , gold_label) :
    acc = np.mean(np.array(predict) == np.array(gold_label))
    return acc

def get_train_data_with_pseudo_label(encoder : ZeroTC , train_data: DataLoader , label_names: list) :
    encoder.eval() 
    label_embedding = encoder(label_names)
    predict = []
    dataset = []
    labels = []
    for data , _ in tqdm(train_data):
        data_embedding = encoder(data)
        scores = torch.matmul(data_embedding , label_embedding.transpose(1 , 0))
        softmax = F.softmax(scores, dim=-1)
        predict.extend(torch.argmax(softmax , dim=-1).cpu().numpy())
        labels.extend(_)
        dataset.append(data)
    logging.info(f'--- get_train_data_with_pseudo_label --- accuracy = {get_accuracy(predict , labels)}')
    pseudo_label_dataset = PseudoLabelDataset(dataset , predict)

    return DataLoader(pseudo_label_dataset , batch_size=train_data.batch_size)

def train(encoder : ZeroTC , train_data : DataLoader, label_names: list) :
    train_data_with_pseudo_label = get_train_data_with_pseudo_label(encoder , train_data ,  label_names)
    import pdb
    pdb.set_trace()
    # 定义损失函数和优化器
    contrastive_loss = ContrastiveLoss()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)

    # 进行自训练循环
    encoder.train()
    epochs = 10
