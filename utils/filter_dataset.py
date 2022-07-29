import argparse
import pandas as pd


def filter_by_avg_vel(data_path, varf_path, 
    lower_bound=None, upper_bound=None, factor='avg_vel'):
    data = pd.read_pickle(data_path)    
    varf = pd.read_pickle(varf_path)
    varf_filtered = varf[varf.metaId.isin(data.metaId.unique())]
    if lower_bound is not None:
        varf_filtered = varf_filtered[varf_filtered[factor] >= lower_bound]
    if upper_bound is not None:
        varf_filtered = varf_filtered[varf_filtered[factor] <= upper_bound]
    data_filtered = data[data.metaId.isin(varf_filtered.metaId.unique())]
    print(f'Before filter: #={data.shape[0]}')
    print(f'After filter: #={data_filtered.shape[0]}')
    new_filename = data_path.replace('.pkl', '_filter.pkl')
    data_filtered.to_pickle(new_filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', default=None, type=str, 
        help='Path to the data to be filtered')
    parser.add_argument('--varf_path', default=None, type=str)

    parser.add_argument('--factor', default='avg_vel', type=str)
    parser.add_argument('--lower_bound', default=None, type=float)
    parser.add_argument('--upper_bound', default=None, type=float)
    args = parser.parse_args()

    filter_by_avg_vel(args.data_path, args.varf_path, 
        args.lower_bound, args.upper_bound, args.factor)
        