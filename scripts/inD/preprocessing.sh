# Prepare images 
mkdir data/inD-dataset-v1.0/images 
mkdir data/inD-dataset-v1.0/images/scene1 data/inD-dataset-v1.0/images/scene2 data/inD-dataset-v1.0/images/scene3 data/inD-dataset-v1.0/images/scene4

cp data/inD-dataset-v1.0/data/00_background.png data/inD-dataset-v1.0/data/07_background.png data/inD-dataset-v1.0/data/18_background.png data/inD-dataset-v1.0/data/30_background.png data/inD-dataset-v1.0/images

mv data/inD-dataset-v1.0/images/00_background.png data/inD-dataset-v1.0/images/scene1/reference.png
mv data/inD-dataset-v1.0/images/07_background.png data/inD-dataset-v1.0/images/scene2/reference.png
mv data/inD-dataset-v1.0/images/18_background.png data/inD-dataset-v1.0/images/scene3/reference.png
mv data/inD-dataset-v1.0/images/30_background.png data/inD-dataset-v1.0/images/scene4/reference.png


# Construct long-term data 
python -m utils.inD_dataset

# Experiment: inD ped to ped, from scenes=2,3,4 to scenes=1 
python -m utils.inD_dataset --reload --labels pedestrian --selected_scenes scene1 --filter_data_dir data/inD-dataset-v1.0/filter/longterm 
python -m utils.inD_dataset --reload --labels pedestrian --selected_scenes scene2 scene3 scene4 --filter_data_dir data/inD-dataset-v1.0/filter/longterm 

mv data/inD-dataset-v1.0/filter/longterm/agent_type/scene2__scene3__scene4 data/inD-dataset-v1.0/filter/longterm/agent_type/scene234
rm -r data/inD-dataset-v1.0/filter/longterm/agent_type/scene2 data/inD-dataset-v1.0/filter/longterm/agent_type/scene3 data/inD-dataset-v1.0/filter/longterm/agent_type/scene4

python -m utils.split_dataset --data_dir data/inD-dataset-v1.0/filter/longterm/agent_type/scene1 --data_filename pedestrian.pkl --val_split 20 --test_split 114 --seed 1 

python -m utils.split_dataset --data_dir data/inD-dataset-v1.0/filter/longterm/agent_type/scene234 --data_filename pedestrian.pkl --val_split 0.1 --test_split 0.2 --seed 1 

# Experiment: inD ped to ped, other scene combinations 
python -m utils.inD_dataset --reload --labels pedestrian --selected_scenes scene1 scene2 scene4 --filter_data_dir data/inD-dataset-v1.0/filter/longterm
python -m utils.inD_dataset --reload --labels pedestrian --selected_scenes scene1 scene3 scene4 --filter_data_dir data/inD-dataset-v1.0/filter/longterm 

mv data/inD-dataset-v1.0/filter/longterm/agent_type/scene1__scene2__scene4 data/inD-dataset-v1.0/filter/longterm/agent_type/scene124
mv data/inD-dataset-v1.0/filter/longterm/agent_type/scene1__scene3__scene4 data/inD-dataset-v1.0/filter/longterm/agent_type/scene134

python -m utils.split_dataset --data_dir data/inD-dataset-v1.0/filter/longterm/agent_type/scene2 --data_filename pedestrian.pkl --val_split 80 --test_split 252 --seed 1 

python -m utils.split_dataset --data_dir data/inD-dataset-v1.0/filter/longterm/agent_type/scene3 --data_filename pedestrian.pkl --val_split 80 --test_split 750 --seed 1 

python -m utils.split_dataset --data_dir data/inD-dataset-v1.0/filter/longterm/agent_type/scene134 --data_filename pedestrian.pkl --val_split 0.1 --test_split 0.2 --seed 1 

python -m utils.split_dataset --data_dir data/inD-dataset-v1.0/filter/longterm/agent_type/scene124 --data_filename pedestrian.pkl --val_split 0.1 --test_split 0.2 --seed 1 



# Construct short-term data 
python -m utils.inD_dataset --raw_data_filename data_8_12_2_5fps.pkl --step 10 --window_size 20 --stride 20 --obs_len 8 --labels pedestrian --filter_data_dir data/inD-dataset-v1.0/filter/shortterm 

# Experiment: cars to trucks 
python -m utils.inD_dataset --reload --raw_data_filename data_8_12_2_5fps.pkl --step 10 --window_size 20 --stride 20 --obs_len 8 --labels car truck_bus --selected_scenes scene1

python -m utils.filter_dataset --data_path data/inD-dataset-v1.0/filter/shortterm/agent_type/scene1/car.pkl --varf_path data/inD-dataset-v1.0/data/varf_8_12_2_5fps.pkl --lower_bound 0.2

python -m utils.filter_dataset --data_path data/inD-dataset-v1.0/filter/shortterm/agent_type/scene1/truck_bus.pkl --varf_path data/inD-dataset-v1.0/data/varf_8_12_2_5fps.pkl --lower_bound 0.2

python -m utils.split_dataset --data_dir data/inD-dataset-v1.0/filter/shortterm/agent_type/scene1 --data_filename car_filter.pkl --val_split 0.1 --test_split 0.2 --seed 1 

python -m utils.split_dataset --data_dir data/inD-dataset-v1.0/filter/shortterm/agent_type/scene1 --data_filename truck_bus_filter.pkl --val_split 40 --test_split 97 --seed 1 

# Experiment: pedestrians 
python -m utils.inD_dataset --reload --additional_data_dir data/inD-dataset-v1.0/data --raw_data_dir data/inD-dataset-v1.0/data --raw_data_filename data_8_12_2_5fps.pkl --filter_data_dir data/inD-dataset-v1.0/filter/shortterm --step 10 --window_size 20 --stride 20 --obs_len 8 --labels pedestrian --selected_scenes scene1 scene2 scene3 scene4

mv data/inD-dataset-v1.0/filter/shortterm/agent_type/scene1__scene2__scene3__scene4 data/inD-dataset-v1.0/filter/shortterm/agent_type/scene1234

python -m utils.filter_dataset --data_path data/inD-dataset-v1.0/filter/shortterm/agent_type/scene1/pedestrian.pkl --varf_path data/inD-dataset-v1.0/data/varf_8_12_2_5fps.pkl --lower_bound 0.2 

python -m utils.split_dataset --data_dir data/inD-dataset-v1.0/filter/shortterm/agent_type/scene1 --data_filename pedestrian_filter.pkl --val_split 100 --test_split 524 --seed 1  
