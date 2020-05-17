
dataset_vae=short_yelp
beta=0
lr=0.5

# AE model path: folder created after step 1
vae_model_path='exp_short_yelp_beta/short_yelp_lr0.5_beta0.0_drop0.5_/model.pt'
kl_start=0
warm_up=10
target_kl=8
fb=2

# VAE model path: folder created after step 2
reconstruct_from='exp_short_yelp_load/short_yelp_warm10_kls0.0_fbdim_tr8.0/model.pt'
decoding_strategy=sample

#WestClass
dataset=yelp
sup_source=docs
model=cnn
with_evaluation=True

# Step 1: Train AE
echo "==================Training AE====================="
python3 vae/text_beta.py --dataset ${dataset_vae} --beta ${beta} --lr ${lr}

# Step 2: Train VAE
echo "\n==================Training VAE====================="
python3 vae/text_anneal_fb.py --dataset ${dataset_vae} --load_path ${vae_model_path} --reset_dec --kl_start ${kl_start} --warm_up ${warm_up} --target_kl ${target_kl} --fb ${fb} --lr ${lr}

# Step 3: Generate pseudo documents
echo "\n==================Generating pseudo-documents====================="
python3 vae/text_anneal_fb.py --dataset ${dataset_vae} --reconstruct_from ${reconstruct_from} --decoding_strategy ${decoding_strategy}
echo "\nGenerated Documents can be found in ./pseudo_documents.txt"

# Step 4: Perform pre- and self-training
echo "\n==================WestClass Classification====================="
python3 westclass/main.py --dataset ${dataset} --sup_source ${sup_source} --model ${model} --with_evaluation ${with_evaluation}