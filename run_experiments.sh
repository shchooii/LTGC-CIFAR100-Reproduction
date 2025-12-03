#!/bin/bash

# # 1. CE (Baseline)
# echo "Running Experiment 1: CE (Baseline)"
# python3 train_ltgc_cifar100.py \
#     --loss ce \
#     --save_path "ckpt_ce_baseline.pth" > log_ce_baseline.txt 2>&1

# 2. CE + Balanced + Generated
echo "Running Experiment 2: CE + Balanced + Generated"
python3 train_ltgc_cifar100.py \
    --loss ce \
    --use_generated \
    --save_path "ckpt_ce_balanced_gen.pth" > log_ce_balanced_gen2.txt 2>&1

# # 3. Focal (Baseline)
# echo "Running Experiment 3: Focal (Baseline)"
# python3 train_ltgc_cifar100.py \
#     --loss focal \
#     --save_path "ckpt_focal_baseline.pth" > log_focal_baseline.txt 2>&1

# 4. Focal + Balanced + Generated
echo "Running Experiment 4: Focal + Balanced + Generated"
python3 train_ltgc_cifar100.py \
    --loss focal \
    --use_generated \
    --save_path "ckpt_focal_balanced_gen.pth" > log_focal_balanced_gen2.txt 2>&1

# # 5. ASL (Baseline)
# echo "Running Experiment 5: ASL (Baseline)"
# python3 train_ltgc_cifar100.py \
#     --loss asl \
#     --save_path "ckpt_asl_baseline.pth" > log_asl_baseline.txt 2>&1

# 6. ASL + Balanced + Generated
echo "Running Experiment 6: ASL + Balanced + Generated"
python3 train_ltgc_cifar100.py \
    --loss asl \
    --use_generated \
    --save_path "ckpt_asl_balanced_gen.pth" > log_asl_balanced_gen2.txt 2>&1
