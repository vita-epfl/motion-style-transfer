# config 
list_train_seed=(1 2 3 4 5) 
batch_size=10
n_epoch=100
n_early_stop=30
n_round=3
config_filename=sdd_shortterm_train.yaml
steps=20

# model 
network=original

# pretrained model 
pretrained_ckpt=ckpts/sdd__ynet__ped.pt
ckpt_path=ckpts/sdd/ped_to_biker

# data path 
dataset_path=filter/shortterm/agent_type/deathCircle_0/Biker
load_data=predefined
# fine-tune setting 
train_net=encoder
list_position=("0 1 2 3 4")  
list_n_train_batch=(3) 
list_lr=(0.0005)


for train_seed in ${list_train_seed[@]}; do
    for lr in ${list_lr[@]}; do 
        for n_train_batch in ${list_n_train_batch[@]}; do 
            for position in "${list_position[@]}"; do
                python train.py --fine_tune --config_filename $config_filename --seed $train_seed --batch_size $batch_size --n_epoch $n_epoch --n_early_stop $n_early_stop --n_round $n_round --dataset_path $dataset_path --network $network --load_data $load_data --pretrained_ckpt $pretrained_ckpt --train_net $train_net --position $position --ckpt_path $ckpt_path --n_train_batch $n_train_batch --lr $lr --steps $steps 
            done 
        done 
    done 
done