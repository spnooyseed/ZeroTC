from argparse import ArgumentParser
from utils.data_utils import set_seed, sample_data, load_data, get_input_data_dir
from utils.model_utils import ZeroTC
from utils.train_utils import train


def main():
    parser = ArgumentParser()
    parser.add_argument("--encoder_name", default="sup-simcse-roberta-base", type=str)
    parser.add_argument("--dataset", default="agnews", type=str)
    args = parser.parse_args()
    set_seed()
    train_data = sample_data(args.dataset, 200)
    label_names = load_data(get_input_data_dir(args.dataset, f"label_names.txt"))
    model = ZeroTC("sup-simcse-roberta-base")
    train(model, train_data, label_names)


if __name__ == "__main__":
    main()
