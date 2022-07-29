# config 
list_train_seed=(2 3) 
batch_size=10
n_epoch=300
n_early_stop=300
n_round=3
config_filename=inD_shortterm_eval.yaml

# model 
network=original

# pretrained model 
pretrained_ckpt=ckpts/sdd__ynet__ped.pt
ckpt_path=ckpts/inD/sdd_to_inD

# data path 
dataset_path=filter/shortterm/agent_type/scene1/pedestrian_filter_s1_t524  
load_data=predefined

# fine-tune setting 
list_train_net=(mosa_1 mosa_2 mosa_4 mosa_8)
list_position=("0 1 2 3 4")
list_n_train_batch=(2) 
list_lr=(0.001)


for train_seed in ${list_train_seed[@]}; do
    for lr in ${list_lr[@]}; do 
        for n_train_batch in ${list_n_train_batch[@]}; do 
            for train_net in ${list_train_net[@]}; do
                for position in "${list_position[@]}"; do
                    python train.py --fine_tune --config_filename $config_filename --seed $train_seed --batch_size $batch_size --n_epoch $n_epoch --n_early_stop $n_early_stop --n_round $n_round --dataset_path $dataset_path --network $network --load_data $load_data --pretrained_ckpt $pretrained_ckpt --train_net $train_net --position $position --ckpt_path $ckpt_path --n_train_batch $n_train_batch --lr $lr --smooth_val 
                done 
            done 
        done 
    done 
done 
