#!/bin/bash
conda activate edm

dataset='in64'
feature_model='dinov2'  # swav or dinov2
top_folder="/home/shared/generative_models/recombination"
n_patches=16

# Train
python extract_patch_features.py \
  --feature_model $feature_model \
  --dataset=$dataset \
  --n_patches $n_patches \
  --outdir="${top_folder}/embeddings/${dataset}/${feature_model}-np${n_patches}/train" \
  --max_size 100000 ;

# EDM2
for gen_model in 'edm2-img64-xl-0671088' ; do # 'edm2-img64-xs-2147483' 'edm2-img64-s-1073741' 'edm2-img64-m-2147483' 'edm2-img64-l-1073741'
  gen_model_folder="${top_folder}/raw_samples/${dataset}/${gen_model}"
  python extract_patch_features.py \
    --feature_model $feature_model \
    --dataset=$gen_model_folder \
    --n_patches $n_patches \
    --outdir="${top_folder}/embeddings/${dataset}/${feature_model}-np${n_patches}/${gen_model}" \
    --max_size 5000 ;
done

# Visual-VAE
python extract_patch_features.py \
  --feature_model $feature_model \
  --n_patches $n_patches \
  --dataset="${top_folder}/raw_samples/${dataset}/v_vae_m0.0_v0.0/50000_random_classes_m0.0_v0.0.zip" \
  --outdir="${top_folder}/embeddings/${dataset}/${feature_model}-np${n_patches}/v_vae_m0.0_v0.0" \
  --max_size 5000 ;