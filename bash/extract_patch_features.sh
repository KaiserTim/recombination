#!/bin/bash
conda activate edm

dataset='in64'

# EDM2
for source in 'edm2-img64-xs-2147483' 'edm2-img64-s-1073741' 'edm2-img64-m-2147483' 'edm2-img64-l-1073741' 'edm2-img64-xl-0671088'; do
  source_folder="/home/shared/generative_models/recombination/raw_samples/${dataset}/${source}"
  python extract_patch_features.py \
    --feature_model 'swav' \
    --dataset=$source_folder \
    --n_patches 16 \
    --outdir="/home/shared/generative_models/recombination/embeddings/${dataset}/${source}" \
    --max_size 5000 ;
done

# Train
python extract_patch_features.py \
  --feature_model 'swav' \
  --dataset=$dataset \
  --n_patches 16 \
  --outdir=/home/shared/generative_models/recombination/embeddings/${dataset}/train/ \
  --max_size 100000 ;

# Visual-VAE
python extract_patch_features.py \
  --feature_model 'swav' \
  --n_patches 16 \
  --dataset="/home/shared/generative_models/recombination/raw_samples/${dataset}/v_vae_m0.0_v0.0/50000_random_classes_m0.0_v0.0.zip" \
  --outdir="/home/shared/generative_models/recombination/embeddings/${dataset}/v_vae_m0.0_v0.0/" \
  --max_size 5000 ;