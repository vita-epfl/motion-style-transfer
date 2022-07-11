# config 
list_eval_seed=(1) 
batch_size=10
n_round=3
config_filename=sdd_shortterm_eval.yaml

# model 
network=fusion
n_fusion=2

# pretrained model 
ckpts=ckpts/sdd__ynetmod__biker.pt
ckpts_name=OODG

# data path 
dataset_path=filter/shortterm/avg_vel/dc_013/Biker/4_8
load_data=predefined


for eval_seed in ${list_eval_seed[@]}; do
    python test.py --config_filename $config_filename --seed $eval_seed --batch_size $batch_size --n_round $n_round --dataset_path $dataset_path --network $network --n_fusion $n_fusion --load_data $load_data --ckpts $ckpts --ckpts_name $ckpts_name
done 

