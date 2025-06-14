#!/bin/bash
# MNIST DeepBoost 完整对照实验

echo "=== MNIST DeepBoost 对照实验 ==="
echo "开始时间: $(date)"

# 创建结果目录
mkdir -p results
cd results

# 实验1: 基准实验
echo "运行实验1: 基准实验"
../driver --data_set=mnist17 --data_filename=../mnist_1_vs_7.data \
  --tree_depth=2 --num_iter=30 --max_features_per_split=100 \
  --beta=1e-4 --lambda=1e-5 --seed=42 \
  > exp1_baseline.log 2>&1

# 实验2: 树深度实验
echo "运行实验2: 树深度实验"
for depth in 2 3 4; do
  echo "  深度: $depth"
  ../driver --data_set=mnist17 --data_filename=../mnist_1_vs_7.data \
    --tree_depth=$depth --num_iter=50 --max_features_per_split=100 \
    --beta=1e-4 --lambda=1e-5 --seed=42 \
    > exp2_depth_${depth}.log 2>&1
done

# 实验3: 特征采样实验
echo "运行实验3: 特征采样实验"
for features in 50 100 200 0; do
  echo "  特征数: $features"
  if [ $features -eq 0 ]; then
    beta=1e-5; lambda=1e-6  # 全特征时使用更强正则化
  else
    beta=1e-4; lambda=1e-5
  fi
  
  ../driver --data_set=mnist17 --data_filename=../mnist_1_vs_7.data \
    --tree_depth=3 --num_iter=40 --max_features_per_split=$features \
    --beta=$beta --lambda=$lambda --seed=42 \
    > exp3_features_${features}.log 2>&1
done

# 实验4: 正则化实验
echo "运行实验4: 正则化实验"
betas=(1e-5 1e-4 1e-3)
lambdas=(1e-6 1e-5 1e-4)
labels=("weak" "medium" "strong")

for i in {0..2}; do
  echo "  正则化: ${labels[$i]}"
  ../driver --data_set=mnist17 --data_filename=../mnist_1_vs_7.data \
    --tree_depth=3 --num_iter=40 --max_features_per_split=100 \
    --beta=${betas[$i]} --lambda=${lambdas[$i]} --seed=42 \
    > exp4_reg_${labels[$i]}.log 2>&1
done

# 实验5: 迭代次数实验
echo "运行实验5: 迭代次数实验"
for iters in 20 50 100; do
  echo "  迭代次数: $iters"
  ../driver --data_set=mnist17 --data_filename=../mnist_1_vs_7.data \
    --tree_depth=3 --num_iter=$iters --max_features_per_split=100 \
    --beta=1e-4 --lambda=1e-5 --seed=42 \
    > exp5_iters_${iters}.log 2>&1
done

# 实验6: 稳定性实验
echo "运行实验6: 稳定性实验"
for seed in 42 123 456 789 999; do
  echo "  种子: $seed"
  ../driver --data_set=mnist17 --data_filename=../mnist_1_vs_7.data \
    --tree_depth=3 --num_iter=50 --max_features_per_split=100 \
    --beta=1e-4 --lambda=1e-5 --seed=$seed \
    > exp6_seed_${seed}.log 2>&1
done

echo "所有实验完成: $(date)"

