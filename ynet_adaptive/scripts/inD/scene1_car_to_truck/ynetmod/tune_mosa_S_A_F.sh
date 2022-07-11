# config 
list_train_seed=(1) 
batch_size=10
n_epoch=300
n_early_stop=3000
n_round=3
config_filename=inD_shortterm_train.yaml

# model 
network=fusion
n_fusion=2

# pretrained model 
pretrained_ckpt=ckpts/inD__ynetmod__car.pt
ckpt_path=ckpts/inD/scene1_car_to_truck/ynetmod

# data path 
dataset_path=filter/shortterm/agent_type/scene1/truck_bus_filter
load_data=predefined

# fine-tune setting 
list_train_net=(mosa_1)
list_position=("scene motion fusion")  
list_n_train_batch=(2) 
list_lr=(0.001)


for train_seed in ${list_train_seed[@]}; do
    for lr in ${list_lr[@]}; do 
        for n_train_batch in ${list_n_train_batch[@]}; do 
            for train_net in ${list_train_net[@]}; do 
                for position in "${list_position[@]}"; do
                    python train.py --fine_tune --config_filename $config_filename --seed $train_seed --batch_size $batch_size --n_epoch $n_epoch --n_early_stop $n_early_stop --n_round $n_round --dataset_path $dataset_path --network $network --n_fusion $n_fusion --load_data $load_data --pretrained_ckpt $pretrained_ckpt --train_net $train_net --position $position --ckpt_path $ckpt_path --n_train_batch $n_train_batch --lr $lr --smooth_val 
                done 
            done 
        done 
    done 
done