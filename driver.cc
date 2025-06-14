/*
Copyright 2015 Google Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <chrono>
#include <sys/resource.h>
#include <unistd.h>
#include <algorithm>
#include <random>

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "boost.h"
#include "io.h"
#include "types.h"

// 声明FLAGS变量（不是定义）
DECLARE_string(data_filename);
DECLARE_string(data_set);
DEFINE_int32(num_iter, 30, "Number of boosting iterations");
DECLARE_int32(tree_depth);
DECLARE_int32(num_folds);
DECLARE_int32(fold_to_cv);
DECLARE_int32(fold_to_test);
DECLARE_double(beta);
DECLARE_double(lambda);
DECLARE_string(loss_type);
DEFINE_int32(seed, 42, "Random seed");
DECLARE_double(noise_prob);

// 新增的FLAGS定义
DEFINE_string(test_filename, "", "Test data filename (optional, for standard train/test split)");

// 全局随机数生成器
std::mt19937 rng;

void ValidateFlags() {
  CHECK_GE(FLAGS_num_iter, 1);
  CHECK_GE(FLAGS_tree_depth, 1);
  CHECK_GE(FLAGS_num_folds, 1);
  CHECK_GE(FLAGS_fold_to_cv, 0);
  CHECK_LT(FLAGS_fold_to_cv, FLAGS_num_folds);
  CHECK_GE(FLAGS_fold_to_test, 0);
  CHECK_LT(FLAGS_fold_to_test, FLAGS_num_folds);
  CHECK_NE(FLAGS_fold_to_cv, FLAGS_fold_to_test);
  CHECK_GT(FLAGS_beta, 0);
  CHECK_GT(FLAGS_lambda, 0);
  CHECK(FLAGS_loss_type == "exponential" || FLAGS_loss_type == "logistic");
  CHECK_GE(FLAGS_noise_prob, 0.0);
  CHECK_LE(FLAGS_noise_prob, 1.0);
}

void SetSeed(int seed) {
  rng.seed(seed);
}

int main(int argc, char** argv) {
  // 记录开始时间
  auto start_time = std::chrono::high_resolution_clock::now();
  
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  ValidateFlags();
  SetSeed(FLAGS_seed);

  vector<Example> train_examples, cv_examples, test_examples;
  
  // 记录数据读取开始时间
  auto data_start = std::chrono::high_resolution_clock::now();
  
  // 判断使用哪种数据读取方式
  if (!FLAGS_test_filename.empty()) {
    printf("Using standard train/test split...\n");
    // 使用标准训练/测试分割
    ReadDataStandardSplit(&train_examples, &test_examples, 
                         FLAGS_data_filename, FLAGS_test_filename);
    
    // 从训练集中分出验证集（可选）
    if (FLAGS_num_folds > 1) {
      std::shuffle(train_examples.begin(), train_examples.end(), rng);
      int cv_size = train_examples.size() / FLAGS_num_folds;
      cv_examples.assign(train_examples.begin(), train_examples.begin() + cv_size);
      train_examples.erase(train_examples.begin(), train_examples.begin() + cv_size);
      
      printf("Created CV set of size %zu from training data\n", cv_examples.size());
      
      // 重新设置权重
      const float initial_wgt = 1.0 / train_examples.size();
      for (Example& example : train_examples) {
        example.weight = initial_wgt;
      }
    }
  } else {
    printf("Using random split from single file...\n");
    // 使用原来的随机分割方式
    ReadData(&train_examples, &cv_examples, &test_examples);
  }
  
  auto data_end = std::chrono::high_resolution_clock::now();
  auto data_duration = std::chrono::duration_cast<std::chrono::milliseconds>(data_end - data_start);
  
  // 输出数据集信息（保留原有格式）
  printf("=== Dataset Information ===\n");
  printf("Training examples: %zu\n", train_examples.size());
  printf("CV examples: %zu\n", cv_examples.size());
  printf("Test examples: %zu\n", test_examples.size());
  printf("Total examples: %zu\n", train_examples.size() + cv_examples.size() + test_examples.size());
  printf("Features per example: %zu\n", train_examples.empty() ? 0 : train_examples[0].values.size());
  printf("Data loading time: %ld ms\n", data_duration.count());
  
  // 显示标签分布
  if (!train_examples.empty()) {
    int pos_train = 0, neg_train = 0;
    for (const Example& ex : train_examples) {
      if (ex.label == 1) pos_train++;
      else neg_train++;
    }
    
    int pos_test = 0, neg_test = 0;
    for (const Example& ex : test_examples) {
      if (ex.label == 1) pos_test++;
      else neg_test++;
    }
    
    printf("Train label distribution - Positive: %d (%.1f%%), Negative: %d (%.1f%%)\n",
           pos_train, 100.0 * pos_train / train_examples.size(),
           neg_train, 100.0 * neg_train / train_examples.size());
    printf("Test label distribution - Positive: %d (%.1f%%), Negative: %d (%.1f%%)\n",
           pos_test, 100.0 * pos_test / test_examples.size(),
           neg_test, 100.0 * neg_test / test_examples.size());
  }
  
  printf("===========================\n\n");

  Model model;
  
  // 记录训练开始时间
  auto train_start = std::chrono::high_resolution_clock::now();
  
  for (int iter = 1; iter <= FLAGS_num_iter; ++iter) {
    auto iter_start = std::chrono::high_resolution_clock::now();
    
    AddTreeToModel(train_examples, &model);
    
    auto iter_end = std::chrono::high_resolution_clock::now();
    auto iter_duration = std::chrono::duration_cast<std::chrono::milliseconds>(iter_end - iter_start);
    
    float cv_error, test_error, avg_tree_size;
    int num_trees;
    
    // 如果有CV集，评估CV错误，否则显示N/A
    if (!cv_examples.empty()) {
      EvaluateModel(cv_examples, model, &cv_error, &avg_tree_size, &num_trees);
    } else {
      cv_error = -1; // 标记无CV数据
    }
    
    EvaluateModel(test_examples, model, &test_error, &avg_tree_size, &num_trees);
    
    // 获取内存使用情况
    struct rusage usage;
    long memory_kb = 0;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
      memory_kb = usage.ru_maxrss;
#ifdef __APPLE__
      memory_kb /= 1024; // macOS上ru_maxrss是字节，转换为KB
#endif
    }
    
    auto total_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(iter_end - train_start);
    
    // 根据是否有CV数据调整输出格式
    if (cv_error >= 0) {
      printf("Iteration: %d, test error: %g, cv error: %g, "
             "avg tree size: %g, num trees: %d, "
             "iter time: %ld ms, total time: %ld ms, memory: %ld KB\n",
             iter, test_error, cv_error, avg_tree_size, num_trees,
             iter_duration.count(), total_elapsed.count(), memory_kb);
    } else {
      printf("Iteration: %d, test error: %g, cv error: N/A, "
             "avg tree size: %g, num trees: %d, "
             "iter time: %ld ms, total time: %ld ms, memory: %ld KB\n",
             iter, test_error, avg_tree_size, num_trees,
             iter_duration.count(), total_elapsed.count(), memory_kb);
    }
  }
  
  // 输出总体统计（保留原有格式）
  auto end_time = std::chrono::high_resolution_clock::now();
  auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  
  printf("\n=== Training Summary ===\n");
  printf("Total training time: %ld ms (%.2f seconds)\n", 
         total_duration.count(), total_duration.count() / 1000.0);
  printf("Final model trees: %zu\n", model.size());
  
  // 计算最终准确率
  float final_test_error, final_avg_tree_size;
  int final_num_trees;
  EvaluateModel(test_examples, model, &final_test_error, &final_avg_tree_size, &final_num_trees);
  printf("Final test accuracy: %.2f%%\n", (1.0 - final_test_error) * 100.0);
  
  // 与基准对比
  printf("\n=== Benchmark Comparison ===\n");
  printf("DeepBoost:       %.2f%% accuracy (%.2f%% error)\n", 
         (1.0 - final_test_error) * 100.0, final_test_error * 100.0);
  printf("NBTree:          85.90%% accuracy (14.10%% error) [benchmark]\n");
  printf("FSS Naive Bayes: 85.95%% accuracy (14.05%% error) [benchmark]\n");
  printf("C4.5-auto:       85.54%% accuracy (14.46%% error) [benchmark]\n");
  
  if (final_test_error < 0.1405) {
    printf("🎉 DeepBoost OUTPERFORMS the best benchmark!\n");
  } else if (final_test_error < 0.1410) {
    printf("✅ DeepBoost matches top-tier performance!\n");
  } else if (final_test_error < 0.1446) {
    printf("👍 DeepBoost performs well compared to benchmarks.\n");
  }
  
  printf("========================\n");
  
  return 0;
}