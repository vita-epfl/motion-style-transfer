import argparse
from utils.data_utils import split_train_val_test_randomly


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default=None, type=str, 
        help='Path to the raw data, can be a subset of the entire dataset')
    parser.add_argument('--data_filename', default=None, type=str)

    parser.add_argument('--val_split', default=None, type=float)
    parser.add_argument('--test_split', default=None, type=float)
    parser.add_argument('--seed', default=1, type=int)

    args = parser.parse_args()

    split_train_val_test_randomly(args.data_dir, args.data_filename, 
        args.val_split, args.test_split, args.seed)