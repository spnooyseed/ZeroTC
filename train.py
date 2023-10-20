from argparse import ArgumentParser
from utils.data_utils import set_seed , sample_data , augment_data

def main() :
    set_seed()
    train_data = sample_data('agnews' , 200)


if __name__ == '__main__' :

    parser = ArgumentParser()
    args = parser.parse_args()
    # import pdb
    # pdb.set_trace()
    main()