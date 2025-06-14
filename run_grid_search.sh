#!/bin/bash

cd /mnt/e/25springcourse/MachineLearning/deepboost || exit 1

BETA_LIST=(0.015625 0.03125 0.0625 0.125 0.25 0.5 1)
LAMBDA_LIST=(0.0001 0.005 0.01 0.05 0.1 0.5)
DEPTH_LIST=(1 2 3 4 5 6)

for beta in "${BETA_LIST[@]}"; do
  for lambda in "${LAMBDA_LIST[@]}"; do
    for depth in "${DEPTH_LIST[@]}"; do
      echo "Running with beta=$beta, lambda=$lambda, tree_depth=$depth"

      # 修改 tree.cc 中的参数
      sed -i "24s/DEFINE_double(beta, .*,/DEFINE_double(beta, $beta,/" tree.cc
      sed -i "25s/DEFINE_double(lambda, .*,/DEFINE_double(lambda, $lambda,/" tree.cc
      sed -i "27s/DEFINE_int32(tree_depth, .*,/DEFINE_int32(tree_depth, $depth,/" tree.cc

      # 编译并运行
      make clean && make driver
      #if [ $? -ne 0 ]; then
      #  echo "Build failed for beta=$beta, lambda=$lambda, depth=$depth" >> run_log.txt
      #  continue
      #fi

      ./driver > "ionosphere/output_beta_${beta}_lambda_${lambda}_depth_${depth}.log"
    done
  done
done

echo "所有组合运行完毕！"
