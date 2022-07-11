import parser
from utils.data_utils import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--additional_data_dir', default='data/sdd/raw', type=str, 
        help='Path to the scene images and variation factor file')
    parser.add_argument('--raw_data_dir', default=None, type=str, 
        help='Path to the raw data, can be a subset of the entire dataset')
    parser.add_argument('--raw_data_filename', default=None, type=str)
    parser.add_argument('--varf_path', default=None, type=str)
    parser.add_argument("--obs_len", default=8, type=int)

    args = parser.parse_args()
    print(args)

    # load raw dataset
    df = pd.read_pickle(os.path.join(args.raw_data_dir, args.raw_data_filename))
    print('Loaded raw dataset')

    # varf_list = ['avg_vel', 'max_vel', 'avg_acc', 'max_acc', 
    #             'abs+max_acc', 'abs+avg_acc', 'min_dist', 'avg_den100', 'avg_den50']
    varf_list = ['avg_vel']

    # get variation factor table 
    df_varfs = get_varf_table(df, varf_list, args.obs_len)
    if args.varf_path is None:
        out_path = os.path.join(args.additional_data_dir, "df_varfs.pkl")
    else: 
        out_path = args.varf_path 
    df_varfs.to_pickle(out_path)
    print(f'Saved variation factor data to {out_path}')
