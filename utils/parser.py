import argparse

__all__ = ['get_parser']


def get_data_args(parser):
    parser.add_argument("--dataset_path", default=None, type=str)
    parser.add_argument('--ckpt_path', default='ckpts')
    parser.add_argument("--shuffle", action='store_true')
    parser.add_argument("--augment", action='store_true')
    parser.add_argument('--load_data', default='sequential', choices=['sequential', 'predefined'])
    parser.add_argument('--show_details', action='store_true')

    # Arguments work for load_data == 'sequential'
    parser.add_argument("--val_split", default=0.1, type=float)
    parser.add_argument("--test_splits", default=None, type=int, nargs='+', 
        help='The number of data left for testing for each val_file')
    parser.add_argument("--val_files", default=None, type=str, nargs="+")
    parser.add_argument("--share_val_test", action='store_true')

    return parser 


def get_model_args(parser):
    # either
    parser.add_argument("--ckpts", default=None, type=str, nargs='+')
    parser.add_argument("--ckpts_name", default=None, type=str, nargs='+')
    # or 
    parser.add_argument("--pretrained_ckpt", default=None, type=str)
    parser.add_argument("--tuned_ckpt", default=None, type=str)
    parser.add_argument('--tuned_ckpts', default=None, type=str, nargs='+')
    
    # added net 
    parser.add_argument('--network', choices=['original', 'embed', 'fusion'])
    parser.add_argument('--n_fusion', default=None, type=int)
    parser.add_argument('--swap_semantic', action='store_true')
    parser.add_argument('--position', default=[], type=str, nargs='+')
    parser.add_argument('--ynet_bias', action='store_true')
    parser.add_argument("--train_net", default="train", type=str, 
        help="Train which part of the network")
    return parser


def get_general_args(parser):
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--gpu", default=None, type=int, help='gpu id to use')
    parser.add_argument("--n_round", default=1, type=int, help='number of rounds in stochastics eval process')  
    parser.add_argument("--config_filename", default=None, type=str)
    return parser


def get_train_args(parser):
    parser.add_argument("--fine_tune", action="store_true")
    parser.add_argument("--n_epoch", default=100, type=int)
    parser.add_argument('--n_early_stop', default=300, type=int)
    parser.add_argument("--n_train_batch", default=None, type=float, help="Limited number of batches for each training agent (fine-tuning), None means no limit (training)")
    # learning rate 
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--steps", default=[], type=int, nargs='+')
    parser.add_argument("--lr_decay_ratio", default=0.1)
    parser.add_argument('--init_check', action='store_true')

    parser.add_argument('--window_size', default=9, type=int)
    parser.add_argument('--smooth_val', action='store_true')

    # Arguments work for load_data == 'sequential'
    parser.add_argument("--train_files", default=None, type=str, nargs="+")
    return parser



def get_parser(is_train):
    parser = argparse.ArgumentParser()
    parser = get_data_args(parser)
    parser = get_model_args(parser)
    parser = get_general_args(parser)
    if is_train:
        parser = get_train_args(parser)
    return parser