import numpy as np
import random
import torch
import logging
from .config import get_input_data_dir
from .model_utils import load_model , load_tokenizer
from tqdm import tqdm
from torch.utils.data import Dataset , DataLoader

class TextDataset(Dataset) :
    def __init__(self , dataset , dataset_prefix , sample_size=10000) :
        super(TextDataset).__init__()
        self.sample_size = sample_size
        self.x = load_data(get_input_data_dir(dataset , f'{dataset_prefix}.txt'))
        self.y = load_data(get_input_data_dir(dataset , f'{dataset_prefix}_labels.txt'))
        self.augment_x = augment_data(self.x)
        self.shuffle_array = np.arange(len(self.x))
        np.random.shuffle(self.shuffle_array)

    def __len__(self):
        return min(len(self.x) , self.sample_size)

    def __getitem__(self, idx) :
        self.idx = self.shuffle_array[idx]
        return self.augment_x[self.idx] , self.y[self.idx]


def load_data(text_path, encoding='utf-8'):

    with open(text_path, encoding=encoding) as f:
        texts = f.readlines()
    return [t.strip() for t in texts]


def save_data(filename, data):

    with open(filename, 'w') as f:
        for d in data:
            f.write(str(d) + '\n')
    f.close()


def set_seed(seed=100) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1

def sample_data(dataset : str , sample_size=10000):
    logging.info(f'-- begin sample_data from {dataset} --')
    train_set = TextDataset('agnews' , 'train' , sample_size)
    train_data_loader = DataLoader(train_set , batch_size=32 , shuffle=True)
    return train_data_loader


def make_prompt(text, instruction):

    words = text.split()
    if len(words) > 100:
        text = ' '.join(words[:100])

    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{text}

### Response:"""


@torch.no_grad()
def augment_data(data : list , batch_size=32) -> np.array :
    # with torch.cuda.amp.autocast():
    # data = np.array(data)
    model_name = 'alpaca-native'
    model = load_model(model_name)
    tokenizer = load_tokenizer(model_name)
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    import pdb
    pdb.set_trace()
    for i in tqdm(range(0 , len(data) , batch_size)) :
        inputs = tokenizer(data[i:i + batch_size] , return_tensors='pt' , padding='max_length', truncation=True , max_length=64)
        seq_len = inputs['input_ids'].shape
        # gpt(inputs , num_return_sequences=3, no_repeat_ngram_size=2, top_k=50)
        output_ids = model.generate(inputs , max_length= 64 + seq_len[1], min_length= 16 + seq_len[1], num_return_sequences=3, no_repeat_ngram_size=2, top_k=50)
        output_ids = output_ids[: , seq_len:]
        gen_text = tokenizer.batch_decode(output_ids , skip_special_tokens=True)

        print(output_ids)