# Get zip file from drive
pip install gdown && gdown https://drive.google.com/uc?id=1XrWBvJj8RJcnVPxTuHWCbG3A9jXRFk8G

# Extract contents
unzip data_checkpoints_SDD_inD_L5.zip && rm data_checkpoints_SDD_inD_L5.zip

# Shift data
unzip data_checkpoints_SDD_inD_L5/data.zip 

# Shift model to checkpoint file
unzip data_checkpoints_SDD_inD_L5/ckpts.zip 

# Remove 
rm -r data_checkpoints_SDD_inD_L5/

# Create shortterm folder (ignore longterm folder)
cd data/sdd/filter
mkdir shortterm
mv agent_type/ avg_vel/ shortterm
cd ../../../
