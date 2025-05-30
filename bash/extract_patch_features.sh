#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

#resolution=64
#feature_model='lpips-vgg'  # Options: swav, dinov2, lpips-alex, lpips-vgg, lpips-squeeze
top_folder="/home/shared/generative_models/recombination"
for resolution in 64 ; do
  dataset="in${resolution}"
  feature_models='dinov2'
  for feature_model in $feature_models ; do
    for n_patches in 16 ; do
#      for factor in 500 ; do
#        # Train
#        python extract_patch_features.py \
#          --feature_model $feature_model \
#          --dataset $dataset \
#          --resolution $resolution \
#          --n_patches $n_patches \
#          --outdir "${top_folder}/embeddings/${dataset}/${feature_model}-np${n_patches}/train" \
#          --max_size $(( $factor * 1024 )) ;
#      done

    # EDM2
    for gen_model in 'edm2-img64-xl-0671088' ; do # IN64: 'edm2-img64-xs-2147483' 'edm2-img64-s-1073741' 'edm2-img64-m-2147483' 'edm2-img64-l-1073741'
#    for gen_model in 'edm2-img512-xl-1342177' ; do # IN512: 'edm2-img512-xs-2147483' 'edm2-img512-s-2147483' 'edm2-img512-m-2147483' 'edm2-img512-l-1879048' 'edm2-img512-xl-1342177'
      gen_model_folder="${top_folder}/raw_samples/${dataset}/${gen_model}"
      python extract_patch_features.py \
        --feature_model $feature_model \
        --dataset $gen_model_folder \
        --resolution $resolution \
        --n_patches $n_patches \
        --outdir "${top_folder}/embeddings/${dataset}/${feature_model}-np${n_patches}/${gen_model}" \
        --max_size $(( 5 * 1024 )) ;

#    # Visual-VAE
#    python extract_patch_features.py \
#      --feature_model $feature_model \
#      --n_patches $n_patches \
#      --resolution $resolution \
#      --dataset "${top_folder}/raw_samples/${dataset}/v_vae_m0.0_v0.0/50000_random_classes_m0.0_v0.0.zip" \
#      --outdir "${top_folder}/embeddings/${dataset}/${feature_model}-np${n_patches}/v_vae_m0.0_v0.0" \
#      --max_size $(( 5 * 1024 )) ;
    done
   done
  done
done
