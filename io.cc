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

#include "io.h"

#include <algorithm>
#include <fstream>
#include <random>

#include "gflags/gflags.h"
#include "glog/logging.h"

DEFINE_string(data_set, "mnist17",
              "Name of data set. Required: One of breastcancer, wpbc, mnist17, ionosphere, "
              "ocr17, ocr49, ocr17-mnist, ocr49-mnist, diabetes, german.");
DEFINE_string(data_filename, "./testdata/mnist_1_vs_7.data",
              "Filename containing data. Required: data_filename not empty.");
DEFINE_int32(num_folds, 5,
             "(num_folds - 2)/num_folds of data used for training, 1/num_folds "
             "of data used for cross-validation, 1/num_folds of data used for "
             "testing. Required: num_folds >= 3.");
DEFINE_int32(fold_to_cv, 0,
             "Zero-indexed fold used for cross-validation. Required: "
             "0 <= fold_to_cv <= num_folds - 1.");
DEFINE_int32(fold_to_test, 1,
             "Zero-indexed fold used for testing. Required: 0 <= fold_to_test "
             "<= num_folds - 1.");
DEFINE_double(noise_prob, 0,
              "Noise probability. Required: 0 <= noise_prob <= 1.");

static std::mt19937 rng;

void SetSeed(uint_fast32_t seed) { rng.seed(seed); }

void SplitString(const string &text, char sep, vector<string>* tokens) {
  int start = 0, end = 0;
  string token;
  while ((end = text.find(sep, start)) != string::npos) {
    token = text.substr(start, end - start);
    if (!token.empty()) {
      tokens->push_back(token);
    }
    start = end + 1;
  }
  token = text.substr(start);
  if (!token.empty()) {
    tokens->push_back(token);
  }
}

bool ParseLineBreastCancer(const string& line, Example* example) {
  example->values.clear();
  vector<string> values;
  SplitString(line, ',', &values);
  for (int i = 0; i < values.size(); ++i) {
    if (i == 0) {
      continue;  // Skip ID
    } else if (i == values.size() - 1) {
      //LOG(INFO) << "Parsed label: '" << values[i] << "' (ASCII codes: ";
      //for (char c : values[i]) {
      //  LOG(INFO) << static_cast<int>(c) << " ";
      //}
      //LOG(INFO) << ")";
      //LOG(INFO) << "Parsed line: " << line;
      //LOG(INFO) << "Label: " << values[i];
      if (values[i] == "2") {  // Benign
        example->label = -1;
      } else if (values[i] == "4") {  // Malignant
        example->label = +1;
      } else {
        LOG(FATAL) << "Unexpected label: " << values[i];
      }
    } else if (values[i] == "?") {
      return false;
    } else {
      float value = atof(values[i].c_str());
      example->values.push_back(value);
    }
  }
  return true;
}

bool ParseLineWpbc(const string& line, Example* example) {
  example->values.clear();
  vector<string> values;
  SplitString(line, ',', &values);
  
  for (int i = 0; i < values.size(); ++i) {
    if (i == 0) {
      continue;  // 跳过ID列
    } else if (i == 1) {
      // 第二列是标签：N = benign(-1), R = malignant(+1)
      if (values[i] == "N") {  // No recurrence (benign)
        example->label = -1;
      } else if (values[i] == "R") {  // Recurrence (malignant)
        example->label = +1;
      } else {
        LOG(FATAL) << "Unexpected label: " << values[i];
      }
    } else {
      // 第3-32列是特征，但需要检查缺失值
      if (values[i] == "?" || values[i].empty()) {
        return false;  // 跳过包含缺失值的样本
      } else {
        float value = atof(values[i].c_str());
        example->values.push_back(value);
      }
    }
  }
  return true;
}

bool ParseLineIon(const string& line, Example* example) {
  example->values.clear();
  vector<string> values;
  SplitString(line, ',', &values);
  for (int i = 0; i < values.size(); ++i) {
    if (i == values.size() - 1) {
      if (values[i] == "b") {  // Bad
        example->label = -1;
      } else if (values[i] == "g") {  // Good
        example->label = +1;
      } else {
        LOG(FATAL) << "Unexpected label: " << values[i];
      }
    } else {
      float value = atof(values[i].c_str());
      example->values.push_back(value);
    }
  }
  return true;
}

bool ParseLineGerman(const string& line, Example* example) {
  example->values.clear();
  vector<string> values;
  SplitString(line, ' ', &values);
  for (int i = 0; i < values.size(); ++i) {
    if (i == values.size() - 1) {
      if (values[i] == "1") {  // Good
        example->label = -1;
      } else if (values[i] == "2") {  // Bad
        example->label = +1;
      } else {
        LOG(FATAL) << "Unexpected label: " << values[i];
      }
    } else {
      float value = atof(values[i].c_str());
      example->values.push_back(value);
    }
  }
  return true;
}

bool ParseLineOcr17(const string& line, Example* example) {
  example->values.clear();
  vector<string> values;
  SplitString(line, ',', &values);
  for (int i = 0; i < values.size(); ++i) {
    if (i == values.size() - 1) {
      if (values[i] == "1") {  // Digit 1
        example->label = -1;
      } else if (values[i] == "7") {  // Digit 7
        example->label = +1;
      } else {
        return false;
      }
    } else {
      float value = atof(values[i].c_str());
      example->values.push_back(value);
    }
  }
  return true;
}

bool ParseLineOcr49(const string& line, Example* example) {
  example->values.clear();
  vector<string> values;
  SplitString(line, ',', &values);
  for (int i = 0; i < values.size(); ++i) {
    if (i == values.size() - 1) {
      if (values[i] == "4") {  // Digit 4
        example->label = -1;
      } else if (values[i] == "9") {  // Digit 9
        example->label = +1;
      } else {
        return false;
      }
    } else {
      float value = atof(values[i].c_str());
      example->values.push_back(value);
    }
  }
  return true;
}

bool ParseLineOcr17Princeton(const string& line, Example* example) {
  example->values.clear();
  vector<string> values;
  SplitString(line, ' ', &values);
  for (int i = 0; i < values.size(); ++i) {
    if (i == values.size() - 1) {
      if (values[i] == "1") {  // Digit 1
        example->label = -1;
      } else if (values[i] == "7") {  // Digit 7
        example->label = +1;
      } else {
        return false;
      }
    } else {
      float value = atof(values[i].c_str());
      example->values.push_back(value);
    }
  }
  return true;
}

bool ParseLineOcr49Princeton(const string& line, Example* example) {
  example->values.clear();
  vector<string> values;
  SplitString(line, ' ', &values);
  for (int i = 0; i < values.size(); ++i) {
    if (i == values.size() - 1) {
      if (values[i] == "4") {  // Digit 4
        example->label = -1;
      } else if (values[i] == "9") {  // Digit 9
        example->label = +1;
      } else {
        return false;
      }
    } else {
      float value = atof(values[i].c_str());
      example->values.push_back(value);
    }
  }
  return true;
}

bool ParseLinePima(const string& line, Example* example) {
  example->values.clear();
  vector<string> values;
  SplitString(line, ',', &values);
  for (int i = 0; i < values.size(); ++i) {
    if (i == values.size() - 1) {
      if (values[i] == "0") {
        example->label = -1;
      } else if (values[i] == "1") {
        example->label = +1;
      } else {
        LOG(FATAL) << "Unexpected label: " << values[i];
      }
    } else {
      float value = atof(values[i].c_str());
      example->values.push_back(value);
    }
  }
  return true;
}

// 在io.cc中添加ParseLineMnist函数
bool ParseLineMnist(const string& line, Example* example) {
  example->values.clear();
  vector<string> values;
  SplitString(line, ',', &values);
  
  // 检查数据格式：应该有785列（784特征 + 1标签）
  if (values.size() != 785) {
    LOG(WARNING) << "Invalid MNIST line format, expected 785 values, got " 
                 << values.size();
    return false;
  }
  
  // 解析784个特征值（前784列）
  example->values.reserve(784);
  for (int i = 0; i < 784; ++i) {
    float pixel_value = atof(values[i].c_str());
    // 归一化像素值到[0,1]范围
    example->values.push_back(pixel_value / 255.0);
  }
  
  // 解析标签（最后一列）
  int label = atoi(values[784].c_str());
  if (label == 0) {
    example->label = -1;  // 数字1 -> -1
  } else if (label == 1) {
    example->label = +1;  // 数字7 -> +1
  } else {
    LOG(WARNING) << "Invalid label: " << label;
    return false;
  }
  
  example->weight = 1.0;
  return true;
}

void ReadData(vector<Example>* train_examples,
              vector<Example>* cv_examples,
              vector<Example>* test_examples) {
  train_examples->clear();
  cv_examples->clear();
  test_examples->clear();
  vector<Example> examples;
  std::ifstream file(FLAGS_data_filename);
  CHECK(file.is_open());
  string line;
  while (!std::getline(file, line).eof()) {
    Example example;
    bool keep_example;
    if (FLAGS_data_set == "breastcancer") {
      keep_example = ParseLineBreastCancer(line, &example);
    } else if (FLAGS_data_set == "wpbc") {  // 添加这个分支
      keep_example = ParseLineWpbc(line, &example);
    } else if (FLAGS_data_set == "mnist17") {  // 新添加
      keep_example = ParseLineMnist(line, &example);
    } else if (FLAGS_data_set == "ionosphere") {
      keep_example = ParseLineIon(line, &example);
    } else if (FLAGS_data_set == "german") {
      keep_example = ParseLineGerman(line, &example);
    } else if (FLAGS_data_set == "ocr17-mnist") {
      keep_example = ParseLineOcr17(line, &example);
    } else if (FLAGS_data_set == "ocr49-mnist") {
      keep_example = ParseLineOcr49(line, &example);
    } else if (FLAGS_data_set == "ocr17") {
      keep_example = ParseLineOcr17Princeton(line, &example);
    } else if (FLAGS_data_set == "ocr49") {
      keep_example = ParseLineOcr49Princeton(line, &example);
    } else if (FLAGS_data_set == "diabetes") {
      keep_example = ParseLinePima(line, &example);
    } else {
      LOG(FATAL) << "Unknown data set: " << FLAGS_data_set;
    }
    if (keep_example) examples.push_back(example);
  }
  std::shuffle(examples.begin(), examples.end(), rng);
  std::uniform_real_distribution<double> dist;
  int fold = 0;
  // TODO(usyed): Two loops is inefficient
  for (Example& example : examples) {
    double r = dist(rng);
    if (r < FLAGS_noise_prob) {
      example.label = -example.label;
    }
    if (fold == FLAGS_fold_to_test) {
      test_examples->push_back(example);
    } else if (fold == FLAGS_fold_to_cv) {
      cv_examples->push_back(example);
    } else {
      train_examples->push_back(example);
    }
    ++fold;
    if (fold == FLAGS_num_folds) fold = 0;
  }
  const float initial_wgt = 1.0 / train_examples->size();
  // TODO(usyed): Three loops is _really_ inefficient
  for (Example& example : *train_examples) {
    example.weight = initial_wgt;
  }

  // 在return;之前添加这些代码
  LOG(INFO) << "Dataset statistics:";
  LOG(INFO) << "Train: " << train_examples->size() << " examples";
  LOG(INFO) << "CV: " << cv_examples->size() << " examples";
  LOG(INFO) << "Test: " << test_examples->size() << " examples";
  
  if (!train_examples->empty()) {
    LOG(INFO) << "Feature dimension: " << (*train_examples)[0].values.size();
    
    // 统计标签分布
    int train_pos = 0, train_neg = 0;
    int cv_pos = 0, cv_neg = 0;
    int test_pos = 0, test_neg = 0;
    
    for (const auto& ex : *train_examples) {
      if (ex.label == 1) train_pos++; else train_neg++;
    }
    for (const auto& ex : *cv_examples) {
      if (ex.label == 1) cv_pos++; else cv_neg++;
    }
    for (const auto& ex : *test_examples) {
      if (ex.label == 1) test_pos++; else test_neg++;
    }
    
    LOG(INFO) << "Label distribution - Train: +" << train_pos << " / -" << train_neg;
    LOG(INFO) << "Label distribution - CV: +" << cv_pos << " / -" << cv_neg;
    LOG(INFO) << "Label distribution - Test: +" << test_pos << " / -" << test_neg;
  }

  return;
}
