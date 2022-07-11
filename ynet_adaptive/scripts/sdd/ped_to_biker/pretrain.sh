# config 
list_train_seed=(1) 
batch_size=10
n_epoch=100
n_early_stop=5
n_round=3
config_filename=sdd_shortterm_train.yaml
ckpt_path=ckpts

# model 
network=original
train_net=train 

# data path 
dataset_path=filter/shortterm/agent_type
train_files=Pedestrian.pkl
val_files=Pedestrian.pkl
val_split=0.1
test_splits=1500
load_data=sequential


for train_seed in ${list_train_seed[@]}; do
    python train.py --config_filename $config_filename --seed $train_seed --batch_size $batch_size --n_epoch $n_epoch --n_early_stop $n_early_stop --n_round $n_round --dataset_path $dataset_path --network $network --load_data $load_data --train_files $train_files --val_files $val_files --val_split $val_split --test_splits $test_splits --train_net $train_net --ckpt_path $ckpt_path --augment 
done 
