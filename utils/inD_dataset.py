import os
import argparse
import numpy as np
import pandas as pd

from utils.data_utils import downsample, filter_short_trajectories, sliding_window, get_varf_table, \
	create_dataset_given_range, create_dataset_by_agent_type, compute_distance_with_neighbors


def load_raw_inD(path='inD-dataset-v1.0/data', scenes=[1], recordings=None):
	'''
	Loads data from inD Dataset. Makes the following preprocessing:
	-filter out unnecessary columns
	-filter out non-pedestrian
	-makes new unique ID (column 'metaId') since original dataset resets id for each scene
	-add scene name to column for visualization
	-output has columns=['trackId', 'frame', 'x', 'y', 'sceneId', 'metaId']
	:param path: str - path to folder, default is 'data/inD'
	:param scenes: list of integers - scenes to load
	:param recordings: list of strings - alternative to scenes, load specified recordings instead, overwrites scenes
	:return: DataFrame containing all trajectories from split
	'''

	scene2rec = {1: ['00', '01', '02', '03', '04', '05', '06'],
				 2: ['07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17'],
				 3: ['18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29'],
				 4: ['30', '31', '32']}

	rec_to_load = []
	for scene in scenes:
		rec_to_load.extend(scene2rec[scene])
	if recordings is not None:
		rec_to_load = recordings
	data = []
	for rec in rec_to_load:
		# load csv
		track = pd.read_csv(os.path.join(path, '{}_tracks.csv'.format(rec)))
		track = track.drop(columns=['trackLifetime', 'heading', 'width', 'length', 'xVelocity', 'yVelocity',
									'xAcceleration', 'yAcceleration', 'lonVelocity', 'latVelocity',
									'lonAcceleration', 'latAcceleration'])
		track_meta = pd.read_csv(os.path.join(path, '{}_tracksMeta.csv'.format(rec)))

		# Filter non-pedestrians
		# pedestrians = track_meta[track_meta['class'] == 'pedestrian']
		# track = track[track['trackId'].isin(pedestrians['trackId'])]
		track_temp = track_meta.set_index('trackId')
		track['label'] = [track_temp.loc[i]['class'] for i in track['trackId']]

		track['rec&trackId'] = [str(recId) + '_' + str(trackId).zfill(6) for recId, trackId in
								zip(track.recordingId, track.trackId)]
		track['sceneId'] = rec
		track['yCenter'] = -track['yCenter']

		# Filter all trajectories outside the scene frame, ie negative values
		track = track[(track['yCenter'] >= 0) & (track['xCenter'] >= 0)]

		data.append(track)

	data = pd.concat(data, ignore_index=True)

	rec_trackId2metaId = {}
	for i, j in enumerate(data['rec&trackId'].unique()):
		rec_trackId2metaId[j] = i
	data['metaId'] = [rec_trackId2metaId[i] for i in data['rec&trackId']]
	data = data.drop(columns=['rec&trackId', 'recordingId'])
	data = data.rename(columns={'xCenter': 'x', 'yCenter': 'y'})

	cols_order = ['trackId', 'frame', 'x', 'y', 'sceneId', 'metaId', 'label']
	data = data.reindex(columns=cols_order)
	return data


def load_and_window_inD(step, window_size, stride, scenes=[1,2,3,4], path='inD-dataset-v1.0/data'):
	"""
	Helper function to aggregate loading and preprocessing in one function. Preprocessing contains:
	- Split fragmented trajectories
	- Downsample fps
	- Filter short trajectories below threshold=window_size
	- Sliding window with window_size and stride
	:param step (int): downsample factor, step=25 means 1fps on inD
	:param window_size (int): Timesteps for one window
	:param stride (int): How many timesteps to stride in windowing. If stride=window_size then there is no overlap
	:param scenes (list of int): Which scenes to load, inD has 4 scenes
	:return pd.df: DataFrame containing the preprocessed data
	"""
	rec2scene = {'00': 'scene1', '01': 'scene1', '02': 'scene1', '03': 'scene1', '04': 'scene1', '05': 'scene1',
				 '06': 'scene1',
				 '07': 'scene2', '08': 'scene2', '09': 'scene2', '10': 'scene2', '11': 'scene2', '12': 'scene2',
				 '13': 'scene2', '14': 'scene2', '15': 'scene2', '16': 'scene2', '17': 'scene2',
				 '18': 'scene3', '19': 'scene3', '20': 'scene3', '21': 'scene3', '22': 'scene3', '23': 'scene3',
				 '24': 'scene3', '25': 'scene3', '26': 'scene3', '27': 'scene3', '28': 'scene3', '29': 'scene3',
				 '30': 'scene4', '31': 'scene4', '32': 'scene4'}

	df = load_raw_inD(path=path, scenes=scenes, recordings=None)
	# inD is already perfectly continuous
	df = downsample(df, step=step)
	df = filter_short_trajectories(df, threshold=window_size)
	df = sliding_window(df, window_size=window_size, stride=stride)
	df['recId'] = df['sceneId'].copy()
	df['sceneId'] = df['recId'].map(rec2scene)

	# To scale inD x and y values into pixel coordinates, one has to divide scene 1 with 0.00814 * 12 and the others with 0.0127 * 12
	# Values from the XX_recordingMeta.csv "orthoPxToMeter" * 12
	df.x /= np.where(df.sceneId == 'scene1', 0.0127 * 12, 0.00814 * 12)
	df.y /= np.where(df.sceneId == 'scene1', 0.0127 * 12, 0.00814 * 12)

	return df


if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument('--additional_data_dir', default='data/inD-dataset-v1.0/data', type=str, 
		help='Path to the scene images and variation factor file')
	parser.add_argument('--raw_data_dir', default='data/inD-dataset-v1.0/data', type=str, 
		help='Path to the raw data, can be a subset of the entire dataset')
	parser.add_argument('--raw_data_filename', default='data_5_30_1fps.pkl', type=str)
	parser.add_argument('--filter_data_dir', default='data/inD-dataset-v1.0/filter/longterm', type=str)

	parser.add_argument('--reload', action='store_true')
	parser.add_argument('--statistic_only', action='store_true', 
		help='By default, show the statistics and save the customized dataset. ' + \
			'Set False to show only the statistics of data split.')

	parser.add_argument("--step", default=25, type=int)
	parser.add_argument("--window_size", default=35, type=int)
	parser.add_argument("--stride", default=35, type=int)
	parser.add_argument("--obs_len", default=5, type=int)

	parser.add_argument("--varf", default=['agent_type'], nargs='+',
						help="Variation factors from: 'avg_vel', 'max_vel', "+\
							"'avg_acc', 'max_acc', 'abs+max_acc', 'abs+avg_acc', "+\
							"'min_dist', 'avg_den50', 'avg_den100', 'agent_type'")
	parser.add_argument("--varf_ranges", help='range of varation factor to take', 
						default=[(0.25, 0.7), (1, 3)])

	parser.add_argument("--labels", default=['pedestrian'], nargs='+', type=str,
						choices=['truck_bus', 'car', 'pedestrian', 'bicycle'])
	parser.add_argument('--selected_scenes', default=['scene1'], type=str, nargs='+')

	args = parser.parse_args()
	args.labels.sort()
	print(args)

	# ============== load raw dataset ===============
	if not args.reload:
		# ## load raw dataset
		df = load_and_window_inD(args.step, args.window_size, args.stride, scenes=[1, 2, 3, 4], path=args.raw_data_dir)
		print('Loaded raw dataset')
		# possibly add a column of distance with neighbors 
		if 'dist' in args.varf or 'den' in args.varf or np.array(['dist' in f or 'den' in f for f in args.varf]).any():
			out = df.groupby('sceneId').apply(compute_distance_with_neighbors)
			for idx_1st in out.index.get_level_values('sceneId').unique():
				df.loc[out[idx_1st].index, 'dist'] = out[idx_1st].values
			print(f'Added a column of distance with neighbors to df')
		out_path = os.path.join(args.raw_data_dir, args.raw_data_filename)
		df.to_pickle(out_path)
		print(f'Saved data to {out_path}')

		# ## get variation factor table 
		varf_list = ['avg_vel', 'max_acc']
		df_varf = get_varf_table(df, varf_list, args.obs_len)
		out_path = os.path.join(args.additional_data_dir, args.raw_data_filename.replace('data', 'varf'))
		df_varf.to_pickle(out_path)
		print(f'Saved variation factor data to {out_path}')

	else:  # reload = True
		# ## or load from stored pickle
		df = pd.read_pickle(os.path.join(args.raw_data_dir, args.raw_data_filename))
		print('Reloaded raw dataset')

	# ============== create customized dataset ================
	if args.varf is not None:
		if args.varf == ['agent_type']:
			out_dir = os.path.join(args.filter_data_dir, args.varf[0])
			create_dataset_by_agent_type(df, args.labels, out_dir, 
				statistic_only=args.statistic_only, selected_scenes=args.selected_scenes)
		else:
			out_dir = os.path.join(args.filter_data_dir, '__'.join(args.varf), '_'.join(args.labels))
			create_dataset_given_range(df, args.varf, args.varf_ranges, args.labels, 
				out_dir, obs_len=args.obs_len, statistic_only=args.statistic_only)
		print(f'Created dataset: \nVariation factor = {args.varf} \nAgents = {args.labels}')