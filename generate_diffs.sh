#!/bin/bash

first_commit=$(git rev-list --max-parents=0 HEAD)
files=(
"configs/realsr_DiT.yaml"
"datapipe/sen2naip_dataset.py"
"overfit_test.py"
"Test_dataloader.py"
"prepare_datasplit.py"
"test_faceir.sh"
"test_realsr.sh"
"trainer.py"
"main.py"
"models/unet.py"
"models/script_util.py"
"models/gaussian_diffusion.py"
"models/respace.py"
)

mkdir -p full_file_diffs_txt

for file in "${files[@]}"; do
    clean_name=$(echo "$file" | tr '/' '_' | sed 's/.*/\L&/')
    git diff "$first_commit" HEAD -- "$file" > "full_file_diffs_txt/${clean_name}.txt"
done
