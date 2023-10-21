from argparse import ArgumentParser
from utils.data_utils import set_seed , sample_data , augment_data
from utils.model_utils import load_model , load_tokenizer , train

def main() :
    parser = ArgumentParser()
    parser.add_argument('--encoder_name' , default='sup-simcse-roberta-base' , type=str)
    parser.add_argument('--dataset' , default='agnews' , type=str)
    args = parser.parse_args()
    set_seed()
    train_data = sample_data(args.dataset , 200)
    simcse_encoder = load_model(args.encoder_name)
    simcse_tokenizer = load_tokenizer(args.encoder_name)
    train(simcse_encoder , simcse_tokenizer , train_data)

if __name__ == '__main__' :
    main()