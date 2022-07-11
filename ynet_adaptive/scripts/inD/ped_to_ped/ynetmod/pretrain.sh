list_train_seed=(1) 
batch_size=10
n_epoch=300
n_early_stop=3000
n_round=3
config_filename=inD_longterm_train.yaml
ckpt_path=ckpts

# model 
network=fusion
n_fusion=2
train_net=train 

# data path 
dataset_path=filter/longterm/agent_type/scene234/pedestrian
load_data=predefined


for train_seed in ${list_train_seed[@]}; do
    CUDA_VISIBLE_DEVICES=1 python train.py --config_filename $config_filename --seed $train_seed --batch_size $batch_size --n_epoch $n_epoch --n_early_stop $n_early_stop --n_round $n_round --dataset_path $dataset_path --network $network --n_fusion $n_fusion --load_data $load_data --train_net $train_net --ckpt_path $ckpt_path --augment 
done 
