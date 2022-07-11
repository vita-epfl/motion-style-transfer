# config 
list_eval_seed=(1) 
batch_size=10
n_round=3
config_filename=sdd_longterm_eval.yaml

# model 
network=original

# pretrained model 
ckpts=ckpts/inD__ynet__ped.pt
ckpts_name=OODG

# data path 
dataset_path=filter/longterm/agent_type/Pedestrian_filter
load_data=predefined


for eval_seed in ${list_eval_seed[@]}; do
    python test.py --config_filename $config_filename --seed $eval_seed --batch_size $batch_size --n_round $n_round --dataset_path $dataset_path --network $network --load_data $load_data --ckpts $ckpts --ckpts_name $ckpts_name
done 

