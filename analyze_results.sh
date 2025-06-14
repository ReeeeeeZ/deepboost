#!/bin/bash
# 分析实验结果 - 最终修复版

echo "=== MNIST DeepBoost 实验结果分析 ==="
echo "分析时间: $(date)"
echo ""

# 检查results目录
if [ ! -d "results" ]; then
    echo "错误: results 目录不存在，请先运行实验"
    exit 1
fi

cd results

# 提取最佳测试错误率
extract_best_error() {
  local file=$1
  if [ -f "$file" ]; then
    # 匹配 "test error: 0.0445833," 格式
    grep "test error:" "$file" | sed 's/.*test error: \([0-9\.]*\).*/\1/' | sort -n | head -1
  else
    echo "N/A"
  fi
}

# 提取最佳CV错误率  
extract_best_cv() {
  local file=$1
  if [ -f "$file" ]; then
    # 匹配 "cv error: 0.05," 格式
    grep "cv error:" "$file" | sed 's/.*cv error: \([0-9\.]*\).*/\1/' | sort -n | head -1
  else
    echo "N/A"
  fi
}

# 提取最终树数量
extract_final_trees() {
  local file=$1
  if [ -f "$file" ]; then
    # 匹配 "num trees: 11" 格式
    tail -1 "$file" | sed 's/.*num trees: \([0-9]*\).*/\1/' 2>/dev/null || echo "N/A"
  else
    echo "N/A"
  fi
}

# 提取平均树大小
extract_avg_size() {
  local file=$1
  if [ -f "$file" ]; then
    # 匹配 "avg tree size: 7," 格式
    tail -1 "$file" | sed 's/.*avg tree size: \([0-9\.]*\).*/\1/' 2>/dev/null || echo "N/A"
  else
    echo "N/A"
  fi
}

# 计算收敛轮次（最佳错误率首次出现的轮次）
extract_convergence() {
  local file=$1
  if [ -f "$file" ]; then
    local best_error=$(extract_best_error "$file")
    if [ "$best_error" != "N/A" ]; then
      # 查找第一次出现最佳错误率的行号，然后提取Iteration号
      grep -n "test error: $best_error" "$file" | head -1 | sed 's/.*Iteration: \([0-9]*\).*/\1/' 2>/dev/null || echo "N/A"
    else
      echo "N/A"
    fi
  else
    echo "N/A"
  fi
}

# 检查实验是否成功
check_success() {
  local file=$1
  if [ -f "$file" ] && grep -q "test error:" "$file"; then
    echo "✓"
  else
    echo "✗"
  fi
}

# 先测试一下提取函数
echo "测试数据提取功能:"
echo "基准实验最佳测试错误率: $(extract_best_error exp1_baseline.log)"
echo "基准实验最佳CV错误率: $(extract_best_cv exp1_baseline.log)"
echo "基准实验最终树数量: $(extract_final_trees exp1_baseline.log)"
echo "基准实验平均树大小: $(extract_avg_size exp1_baseline.log)"
echo ""

# 生成结果汇总
echo "===================================================================================="
echo "实验结果汇总表"
echo "===================================================================================="
printf "%-20s %-25s %-10s %-15s %-15s %-10s %-10s %-12s\n" \
  "实验组" "配置" "状态" "最佳测试错误率" "最佳CV错误率" "收敛轮次" "树数量" "平均树大小"
echo "------------------------------------------------------------------------------------"

# 实验1: 基准实验
file="exp1_baseline.log"
printf "%-20s %-25s %-10s %-15s %-15s %-10s %-10s %-12s\n" \
  "实验1(基准)" "depth=2,iter=30,feat=100" "$(check_success $file)" \
  "$(extract_best_error $file)" "$(extract_best_cv $file)" \
  "$(extract_convergence $file)" "$(extract_final_trees $file)" "$(extract_avg_size $file)"

# 实验2: 树深度实验
for depth in 2 3 4; do
  file="exp2_depth_${depth}.log"
  printf "%-20s %-25s %-10s %-15s %-15s %-10s %-10s %-12s\n" \
    "实验2(深度)" "depth=$depth,iter=50" "$(check_success $file)" \
    "$(extract_best_error $file)" "$(extract_best_cv $file)" \
    "$(extract_convergence $file)" "$(extract_final_trees $file)" "$(extract_avg_size $file)"
done

# 实验3: 特征采样实验
for features in 50 100 200 0; do
  file="exp3_features_${features}.log"
  feat_label=$([ $features -eq 0 ] && echo "all(784)" || echo "$features")
  printf "%-20s %-25s %-10s %-15s %-15s %-10s %-10s %-12s\n" \
    "实验3(特征)" "feat=$feat_label,depth=3" "$(check_success $file)" \
    "$(extract_best_error $file)" "$(extract_best_cv $file)" \
    "$(extract_convergence $file)" "$(extract_final_trees $file)" "$(extract_avg_size $file)"
done

# 实验4: 正则化实验
for reg in weak medium strong; do
  file="exp4_reg_${reg}.log"
  printf "%-20s %-25s %-10s %-15s %-15s %-10s %-10s %-12s\n" \
    "实验4(正则化)" "reg=$reg,depth=3" "$(check_success $file)" \
    "$(extract_best_error $file)" "$(extract_best_cv $file)" \
    "$(extract_convergence $file)" "$(extract_final_trees $file)" "$(extract_avg_size $file)"
done

# 实验5: 迭代次数实验
for iters in 20 50 100; do
  file="exp5_iters_${iters}.log"
  printf "%-20s %-25s %-10s %-15s %-15s %-10s %-10s %-12s\n" \
    "实验5(迭代)" "iter=$iters,depth=3" "$(check_success $file)" \
    "$(extract_best_error $file)" "$(extract_best_cv $file)" \
    "$(extract_convergence $file)" "$(extract_final_trees $file)" "$(extract_avg_size $file)"
done

echo ""
echo "===================================================================================="
echo "实验6: 稳定性测试结果"
echo "===================================================================================="
printf "%-15s %-15s %-15s %-10s %-10s %-12s\n" \
  "随机种子" "测试错误率" "CV错误率" "收敛轮次" "树数量" "平均树大小"
echo "--------------------------------------------------------------------------------"

# 收集稳定性测试数据
declare -a test_errors=()
declare -a cv_errors=()

for seed in 42 123 456 789 999; do
  file="exp6_seed_${seed}.log"
  test_err=$(extract_best_error $file)
  cv_err=$(extract_best_cv $file)
  
  printf "%-15s %-15s %-15s %-10s %-10s %-12s\n" \
    "$seed" "$test_err" "$cv_err" \
    "$(extract_convergence $file)" "$(extract_final_trees $file)" "$(extract_avg_size $file)"
  
  # 收集有效数值用于统计分析
  if [[ "$test_err" =~ ^[0-9]+\.?[0-9]*$ ]]; then
    test_errors+=($test_err)
  fi
  if [[ "$cv_err" =~ ^[0-9]+\.?[0-9]*$ ]]; then
    cv_errors+=($cv_err)
  fi
done

echo ""
echo "===================================================================================="
echo "统计分析"
echo "===================================================================================="

# 计算稳定性统计
if [ ${#test_errors[@]} -gt 0 ]; then
  # 将数组转换为字符串传递给awk
  test_errors_str=$(printf '%s ' "${test_errors[@]}")
  
  # 使用awk计算统计量
  stats=$(awk -v errors="$test_errors_str" '
  BEGIN {
    n = split(errors, arr, " ")
    sum = 0
    count = 0
    for (i = 1; i <= n; i++) {
      if (arr[i] != "" && arr[i] ~ /^[0-9\.]+$/) {
        sum += arr[i]
        count++
      }
    }
    
    if (count > 0) {
      mean = sum / count
      
      sum_sq = 0
      for (i = 1; i <= n; i++) {
        if (arr[i] != "" && arr[i] ~ /^[0-9\.]+$/) {
          diff = arr[i] - mean
          sum_sq += diff * diff
        }
      }
      std = sqrt(sum_sq / count)
      
      # 找最小最大值
      min = 999; max = 0
      for (i = 1; i <= n; i++) {
        if (arr[i] != "" && arr[i] ~ /^[0-9\.]+$/) {
          if (arr[i] < min) min = arr[i]
          if (arr[i] > max) max = arr[i]
        }
      }
      
      printf "%.6f %.6f %.6f %.6f %d\n", mean, std, min, max, count
    } else {
      printf "N/A N/A N/A N/A 0\n"
    }
  }')
  
  read avg_test std_test min_test max_test count <<< "$stats"
  
  echo "稳定性分析 (测试错误率):"
  echo "  平均值: $avg_test"
  echo "  标准差: $std_test"
  echo "  样本数: $count"
  echo "  范围: $min_test - $max_test"
fi

echo ""

# 找出最佳配置
echo "===================================================================================="
echo "最佳配置推荐"
echo "===================================================================================="

# 扫描所有结果文件找最佳性能
best_test_error=1.0
best_config=""
best_file=""

for file in *.log; do
  if [ -f "$file" ]; then
    test_err=$(extract_best_error "$file")
    if [[ "$test_err" =~ ^[0-9]+\.?[0-9]*$ ]]; then
      # 使用awk进行浮点数比较
      is_better=$(awk -v current="$test_err" -v best="$best_test_error" 'BEGIN{print (current < best) ? 1 : 0}')
      if [ "$is_better" -eq 1 ]; then
        best_test_error=$test_err
        best_file=$file
        
        # 解析配置信息
        case $file in
          exp1_*) best_config="基准配置: depth=2, iter=30, feat=100" ;;
          exp2_depth_*) depth=$(echo $file | sed 's/.*depth_\([0-9]*\).*/\1/'); best_config="最优深度: depth=$depth" ;;
          exp3_features_*) feat=$(echo $file | sed 's/.*features_\([0-9]*\).*/\1/'); best_config="最优特征: feat=$feat" ;;
          exp4_reg_*) reg=$(echo $file | sed 's/.*reg_\([^.]*\).*/\1/'); best_config="最优正则化: $reg" ;;
          exp5_iters_*) iter=$(echo $file | sed 's/.*iters_\([0-9]*\).*/\1/'); best_config="最优迭代: iter=$iter" ;;
          exp6_seed_*) seed=$(echo $file | sed 's/.*seed_\([0-9]*\).*/\1/'); best_config="种子$seed结果" ;;
        esac
      fi
    fi
  fi
done

echo "🏆 最佳性能:"
echo "  测试错误率: $best_test_error"
echo "  配置: $best_config"
echo "  日志文件: $best_file"
echo ""

# 显示前5名结果
echo "📊 性能排行榜 (前5名):"
echo "------------------------"
temp_file=$(mktemp)
for file in *.log; do
  if [ -f "$file" ]; then
    test_err=$(extract_best_error "$file")
    if [[ "$test_err" =~ ^[0-9]+\.?[0-9]*$ ]]; then
      echo "$test_err $file" >> "$temp_file"
    fi
  fi
done

if [ -f "$temp_file" ] && [ -s "$temp_file" ]; then
  sort -n "$temp_file" | head -5 | while read error file; do
    echo "  $error - $file"
  done
  rm "$temp_file"
else
  echo "  无有效结果数据"
  [ -f "$temp_file" ] && rm "$temp_file"
fi

echo ""

# 生成实验报告总结
echo "===================================================================================="
echo "实验报告总结"
echo "===================================================================================="

total_experiments=$(ls -1 *.log 2>/dev/null | wc -l)
successful_experiments=0
for file in *.log; do
  if grep -q "test error:" "$file" 2>/dev/null; then
    ((successful_experiments++))
  fi
done

success_rate=$(awk -v succ="$successful_experiments" -v total="$total_experiments" 'BEGIN{printf "%.1f", succ*100/total}')

echo "📊 实验概况:"
echo "  总实验数: $total_experiments"
echo "  成功实验数: $successful_experiments" 
echo "  成功率: $success_rate%"
echo ""

echo "📁 生成的文件:"
ls -la *.log | awk '{printf "  %-30s %10s bytes  %s %s %s\n", $9, $5, $6, $7, $8}'

echo ""
echo "💡 关键发现:"
echo "  - 基准实验(depth=2)测试错误率: $(extract_best_error exp1_baseline.log)"
echo "  - 深度2实验最佳错误率: $(extract_best_error exp2_depth_2.log)"
echo "  - 深度3实验最佳错误率: $(extract_best_error exp2_depth_3.log)"
echo "  - 深度4实验最佳错误率: $(extract_best_error exp2_depth_4.log)"

echo ""
echo "🔍 详细分析:"
echo "  各实验组的详细训练曲线请查看对应的 .log 文件"
echo "  可使用以下命令查看特定实验的完整训练过程:"
echo "    tail -50 results/<实验文件名>.log"
echo "    grep \"Iteration\" results/<实验文件名>.log"

cd ..
