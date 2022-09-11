# config 
list_eval_seed=(1) 
batch_size=10
n_round=3
config_filename=sdd_shortterm_eval.yaml

# model 
network=fusion
n_fusion=2

# pretrained model 
ckpts=ckpts/sdd__ynetmod__ped2biker.pt
ckpts_name=OODG

# data path 
dataset_path=filter/shortterm/agent_type/deathCircle_0/Biker
load_data=predefined


for eval_seed in ${list_eval_seed[@]}; do
    python test.py --config_filename $config_filename --seed $eval_seed --batch_size $batch_size --n_round $n_round --dataset_path $dataset_path --network $network --n_fusion $n_fusion --load_data $load_data --ckpts $ckpts --ckpts_name $ckpts_name
done 

