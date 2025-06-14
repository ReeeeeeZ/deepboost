#!/bin/bash
# MNIST DeepBoost 完整对照实验

echo "=== MNIST DeepBoost 对照实验 ==="
echo "开始时间: $(date)"

# 检查必要文件
if [ ! -f "./driver" ]; then
    echo "错误: driver 可执行文件不存在"
    exit 1
fi

if [ ! -f "./testdata/mnist_1_vs_7.data" ]; then
    echo "错误: mnist_1_vs_7.data 数据文件不存在"
    exit 1
fi

# 获取绝对路径
DRIVER_PATH=$(pwd)/driver
DATA_PATH=$(pwd)/mnist_1_vs_7.data
RESULTS_DIR=$(pwd)/results

# 创建结果目录
echo "创建结果目录..."
mkdir -p "$RESULTS_DIR"

# 实验1: 基准实验
echo "运行实验1: 基准实验"
$DRIVER_PATH --tree_depth=2 --num_iter=30 --max_features_per_split=100 \
  --beta=1e-4 --lambda=1e-5 --seed=42 \
  > "$RESULTS_DIR/exp1_baseline.log" 2>&1

# 实验2: 树深度实验
echo "运行实验2: 树深度实验"
for depth in 2 3 4; do
  echo "  深度: $depth"
  $DRIVER_PATH --tree_depth=$depth --num_iter=50 --max_features_per_split=100 \
    --beta=1e-4 --lambda=1e-5 --seed=42 \
    > "$RESULTS_DIR/exp2_depth_${depth}.log" 2>&1
  
  # 添加延迟避免冲突
  sleep 1
done

# 实验3: 特征采样实验
echo "运行实验3: 特征采样实验"
for features in 50 100 200; do
  echo "  特征数: $features"
  $DRIVER_PATH --tree_depth=3 --num_iter=40 --max_features_per_split=$features \
    --beta=1e-4 --lambda=1e-5 --seed=42 \
    > "$RESULTS_DIR/exp3_features_${features}.log" 2>&1
  
  sleep 1
done

# 实验3额外: 全特征测试（可能失败）
echo "  特征数: 全部(784)"
$DRIVER_PATH --tree_depth=3 --num_iter=20 --max_features_per_split=0 \
  --beta=1e-5 --lambda=1e-6 --seed=42 \
  > "$RESULTS_DIR/exp3_features_0.log" 2>&1

sleep 1

# 实验4: 正则化实验
echo "运行实验4: 正则化实验"
echo "  正则化: 弱"
$DRIVER_PATH --tree_depth=3 --num_iter=40 --max_features_per_split=100 \
  --beta=1e-5 --lambda=1e-6 --seed=42 \
  > "$RESULTS_DIR/exp4_reg_weak.log" 2>&1

sleep 1

echo "  正则化: 中等"
$DRIVER_PATH --tree_depth=3 --num_iter=40 --max_features_per_split=100 \
  --beta=1e-4 --lambda=1e-5 --seed=42 \
  > "$RESULTS_DIR/exp4_reg_medium.log" 2>&1

sleep 1

echo "  正则化: 强"
$DRIVER_PATH --tree_depth=3 --num_iter=40 --max_features_per_split=100 \
  --beta=1e-3 --lambda=1e-4 --seed=42 \
  > "$RESULTS_DIR/exp4_reg_strong.log" 2>&1

sleep 1

# 实验5: 迭代次数实验
echo "运行实验5: 迭代次数实验"
for iters in 20 50 100; do
  echo "  迭代次数: $iters"
  $DRIVER_PATH --tree_depth=3 --num_iter=$iters --max_features_per_split=100 \
    --beta=1e-4 --lambda=1e-5 --seed=42 \
    > "$RESULTS_DIR/exp5_iters_${iters}.log" 2>&1
  
  sleep 1
done

# 实验6: 稳定性实验
echo "运行实验6: 稳定性实验"
for seed in 42 123 456 789 999; do
  echo "  种子: $seed"
  $DRIVER_PATH --tree_depth=3 --num_iter=50 --max_features_per_split=100 \
    --beta=1e-4 --lambda=1e-5 --seed=$seed \
    > "$RESULTS_DIR/exp6_seed_${seed}.log" 2>&1
  
  sleep 1
done

echo ""
echo "所有实验完成: $(date)"
echo "结果文件保存在 $RESULTS_DIR 目录中"
ls -la "$RESULTS_DIR"/*.log
