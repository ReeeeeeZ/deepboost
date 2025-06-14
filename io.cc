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

DEFINE_string(data_set, "adult",
              "Name of data set. Required: One of adult, breastcancer, wpbc, ionosphere, "
              "ocr17, ocr49, ocr17-mnist, ocr49-mnist, diabetes, german.");
DEFINE_string(data_filename, "./testdata/adult/adult.data",
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

bool ParseLineAdult(const string& line, Example* example) {
  example->values.clear();
  vector<string> values;
  SplitString(line, ',', &values);
  
  if (values.size() != 15) {
    return false; // 期望15列（14个特征 + 1个标签）
  }
  
  // 去除首尾空格的辅助函数
  auto trim = [](string& str) {
    str.erase(0, str.find_first_not_of(" \t"));
    str.erase(str.find_last_not_of(" \t") + 1);
  };
  
  // 去除所有值的首尾空格
  for (string& value : values) {
    trim(value);
  }
  
  // 检查是否有缺失值
  for (const string& value : values) {
    if (value == "?" || value.empty()) {
      return false; // 跳过包含缺失值的样本
    }
  }
  
  try {
    // ========== 数值特征（标准化处理） ==========
    
    // 年龄 (索引0) - 标准化到合理范围
    float age = atof(values[0].c_str());
    example->values.push_back(age / 100.0);
    
    // fnlwgt (索引2) - 对数变换后标准化
    float fnlwgt = atof(values[2].c_str());
    example->values.push_back(log(fnlwgt + 1) / 20.0);
    
    // 教育年限 (索引4) - 标准化
    float education_num = atof(values[4].c_str());
    example->values.push_back(education_num / 20.0);
    
    // 资本收益 (索引10) - 对数变换后标准化
    float capital_gain = atof(values[10].c_str());
    example->values.push_back(log(capital_gain + 1) / 15.0);
    
    // 资本损失 (索引11) - 对数变换后标准化
    float capital_loss = atof(values[11].c_str());
    example->values.push_back(log(capital_loss + 1) / 15.0);
    
    // 每周工作小时 (索引12) - 标准化
    float hours_per_week = atof(values[12].c_str());
    example->values.push_back(hours_per_week / 100.0);
    
    // ========== 分类特征（编码处理） ==========
    
    // 工作类别 (索引1)
    const string& workclass = values[1];
    if (workclass == "Private") {
      example->values.push_back(1.0);
    } else if (workclass == "Self-emp-not-inc") {
      example->values.push_back(2.0);
    } else if (workclass == "Self-emp-inc") {
      example->values.push_back(3.0);
    } else if (workclass == "Federal-gov") {
      example->values.push_back(4.0);
    } else if (workclass == "Local-gov") {
      example->values.push_back(5.0);
    } else if (workclass == "State-gov") {
      example->values.push_back(6.0);
    } else if (workclass == "Without-pay") {
      example->values.push_back(7.0);
    } else if (workclass == "Never-worked") {
      example->values.push_back(8.0);
    } else {
      example->values.push_back(0.0); // 未知类别
    }
    
    // 教育程度 (索引3) - 按教育水平编码
    const string& education = values[3];
    if (education == "Preschool") {
      example->values.push_back(1.0);
    } else if (education == "1st-4th") {
      example->values.push_back(2.0);
    } else if (education == "5th-6th") {
      example->values.push_back(3.0);
    } else if (education == "7th-8th") {
      example->values.push_back(4.0);
    } else if (education == "9th") {
      example->values.push_back(5.0);
    } else if (education == "10th") {
      example->values.push_back(6.0);
    } else if (education == "11th") {
      example->values.push_back(7.0);
    } else if (education == "12th") {
      example->values.push_back(8.0);
    } else if (education == "HS-grad") {
      example->values.push_back(9.0);
    } else if (education == "Some-college") {
      example->values.push_back(10.0);
    } else if (education == "Assoc-voc") {
      example->values.push_back(11.0);
    } else if (education == "Assoc-acdm") {
      example->values.push_back(12.0);
    } else if (education == "Bachelors") {
      example->values.push_back(13.0);
    } else if (education == "Masters") {
      example->values.push_back(14.0);
    } else if (education == "Prof-school") {
      example->values.push_back(15.0);
    } else if (education == "Doctorate") {
      example->values.push_back(16.0);
    } else {
      example->values.push_back(0.0); // 未知类别
    }
    
    // 婚姻状况 (索引5) - 简化为已婚/未婚
    const string& marital_status = values[5];
    if (marital_status == "Married-civ-spouse" || 
        marital_status == "Married-AF-spouse" ||
        marital_status == "Married-spouse-absent") {
      example->values.push_back(1.0); // 已婚
    } else {
      example->values.push_back(0.0); // 未婚
    }
    
    // 职业 (索引6) - 按技能水平编码
    const string& occupation = values[6];
    if (occupation == "Prof-specialty") {
      example->values.push_back(6.0); // 专业技术
    } else if (occupation == "Exec-managerial") {
      example->values.push_back(5.0); // 管理层
    } else if (occupation == "Tech-support") {
      example->values.push_back(4.0); // 技术支持
    } else if (occupation == "Sales") {
      example->values.push_back(3.0); // 销售
    } else if (occupation == "Adm-clerical") {
      example->values.push_back(3.0); // 文职
    } else if (occupation == "Craft-repair") {
      example->values.push_back(2.0); // 技工
    } else if (occupation == "Transport-moving") {
      example->values.push_back(2.0); // 运输
    } else if (occupation == "Machine-op-inspct") {
      example->values.push_back(2.0); // 机械操作
    } else if (occupation == "Other-service") {
      example->values.push_back(1.0); // 其他服务
    } else if (occupation == "Handlers-cleaners") {
      example->values.push_back(1.0); // 清洁工
    } else if (occupation == "Farming-fishing") {
      example->values.push_back(1.0); // 农渔业
    } else if (occupation == "Protective-serv") {
      example->values.push_back(3.0); // 保护服务
    } else if (occupation == "Priv-house-serv") {
      example->values.push_back(1.0); // 家政服务
    } else if (occupation == "Armed-Forces") {
      example->values.push_back(4.0); // 军队
    } else {
      example->values.push_back(0.0); // 未知职业
    }
    
    // 家庭关系 (索引7)
    const string& relationship = values[7];
    if (relationship == "Husband") {
      example->values.push_back(3.0);
    } else if (relationship == "Wife") {
      example->values.push_back(2.0);
    } else if (relationship == "Own-child") {
      example->values.push_back(1.0);
    } else if (relationship == "Not-in-family") {
      example->values.push_back(0.0);
    } else if (relationship == "Other-relative") {
      example->values.push_back(1.0);
    } else if (relationship == "Unmarried") {
      example->values.push_back(0.0);
    } else {
      example->values.push_back(0.0);
    }
    
    // 种族 (索引8) - 简单编码
    const string& race = values[8];
    if (race == "White") {
      example->values.push_back(1.0);
    } else if (race == "Black") {
      example->values.push_back(2.0);
    } else if (race == "Asian-Pac-Islander") {
      example->values.push_back(3.0);
    } else if (race == "Amer-Indian-Eskimo") {
      example->values.push_back(4.0);
    } else if (race == "Other") {
      example->values.push_back(5.0);
    } else {
      example->values.push_back(0.0);
    }
    
    // 性别 (索引9)
    const string& sex = values[9];
    example->values.push_back(sex == "Male" ? 1.0 : 0.0);
    
    // 国家 (索引13) - 简化为美国/非美国
    const string& native_country = values[13];
    example->values.push_back(native_country == "United-States" ? 1.0 : 0.0);
    
    // ========== 标签处理 ==========
    
    // 收入标签 (索引14)
    const string& income = values[14];
    if (income == "<=50K") {
      example->label = -1; // 低收入
    } else if (income == ">50K") {
      example->label = +1; // 高收入
    } else {
      return false; // 未知标签
    }
    
    return true;
    
  } catch (const std::exception& e) {
    // 捕获任何解析错误
    return false;
  } catch (...) {
    // 捕获其他异常
    return false;
  }
}

void ReadData(vector<Example>* train_examples,
              vector<Example>* cv_examples,
              vector<Example>* test_examples) {
  train_examples->clear();
  cv_examples->clear();
  test_examples->clear();
  vector<Example> examples;
  
  std::ifstream file(FLAGS_data_filename);
  CHECK(file.is_open()) << "Cannot open file: " << FLAGS_data_filename;
  
  string line;
  int line_count = 0;
  int parsed_count = 0;
  int skipped_count = 0;
  
  LOG(INFO) << "Reading data from: " << FLAGS_data_filename;
  LOG(INFO) << "Dataset: " << FLAGS_data_set;
  
  while (!std::getline(file, line).eof()) {
    line_count++;
    
    // 跳过空行
    if (line.empty()) {
      skipped_count++;
      continue;
    }
    
    Example example;
    bool keep_example;
    
    if (FLAGS_data_set == "breastcancer") {
      keep_example = ParseLineBreastCancer(line, &example);
    } else if (FLAGS_data_set == "wpbc") {
      keep_example = ParseLineWpbc(line, &example);
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
    } else if (FLAGS_data_set == "adult") {
      keep_example = ParseLineAdult(line, &example);
    } else {
      LOG(FATAL) << "Unknown data set: " << FLAGS_data_set;
    }
    
    if (keep_example) {
      examples.push_back(example);
      parsed_count++;
    } else {
      skipped_count++;
    }
    
    // 每10000行输出一次进度
    if (line_count % 10000 == 0) {
      LOG(INFO) << "Processed " << line_count << " lines, parsed " << parsed_count << " examples";
    }
  }
  
  LOG(INFO) << "=== Data Reading Summary ===";
  LOG(INFO) << "Total lines read: " << line_count;
  LOG(INFO) << "Successfully parsed: " << parsed_count;
  LOG(INFO) << "Skipped (empty/invalid): " << skipped_count;
  LOG(INFO) << "Parse success rate: " << (100.0 * parsed_count / line_count) << "%";
  
  if (examples.empty()) {
    LOG(FATAL) << "No examples were parsed from the data file!";
  }
  
  // 输出特征数量信息
  if (!examples.empty()) {
    LOG(INFO) << "Features per example: " << examples[0].values.size();
    
    // 统计标签分布
    int positive_count = 0, negative_count = 0;
    for (const Example& ex : examples) {
      if (ex.label == 1) positive_count++;
      else if (ex.label == -1) negative_count++;
    }
    LOG(INFO) << "Label distribution - Positive: " << positive_count 
              << " (" << (100.0 * positive_count / examples.size()) << "%), "
              << "Negative: " << negative_count 
              << " (" << (100.0 * negative_count / examples.size()) << "%)";
  }
  
  // 原有的数据分割逻辑保持不变
  std::shuffle(examples.begin(), examples.end(), rng);
  std::uniform_real_distribution<double> dist;
  int fold = 0;
  
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
  for (Example& example : *train_examples) {
    example.weight = initial_wgt;
  }
  
  LOG(INFO) << "=== Data Split Summary ===";
  LOG(INFO) << "Training examples: " << train_examples->size();
  LOG(INFO) << "CV examples: " << cv_examples->size();
  LOG(INFO) << "Test examples: " << test_examples->size();
  LOG(INFO) << "===========================";
  
  return;
}

void ReadDataStandardSplit(vector<Example>* train_examples,
                          vector<Example>* test_examples,
                          const string& train_file,
                          const string& test_file) {
  train_examples->clear();
  test_examples->clear();
  
  // 读取训练数据
  std::ifstream train_file_stream(train_file);
  CHECK(train_file_stream.is_open()) << "Cannot open train file: " << train_file;
  
  string line;
  int train_total = 0, train_parsed = 0, train_skipped = 0;
  
  LOG(INFO) << "Reading training data from: " << train_file;
  while (!std::getline(train_file_stream, line).eof()) {
    train_total++;
    if (line.empty()) {
      train_skipped++;
      continue;
    }
    
    Example example;
    if (ParseLineAdult(line, &example)) {
      train_examples->push_back(example);
      train_parsed++;
    } else {
      train_skipped++;
    }
    
    // 每10000行输出一次进度
    if (train_total % 10000 == 0) {
      LOG(INFO) << "Train: processed " << train_total << " lines, parsed " << train_parsed << " examples";
    }
  }
  
  // 读取测试数据
  std::ifstream test_file_stream(test_file);
  CHECK(test_file_stream.is_open()) << "Cannot open test file: " << test_file;
  
  int test_total = 0, test_parsed = 0, test_skipped = 0;
  
  LOG(INFO) << "Reading test data from: " << test_file;
  while (!std::getline(test_file_stream, line).eof()) {
    test_total++;
    if (line.empty()) {
      test_skipped++;
      continue;
    }
    
    Example example;
    if (ParseLineAdult(line, &example)) {
      test_examples->push_back(example);
      test_parsed++;
    } else {
      test_skipped++;
    }
    
    // 每5000行输出一次进度
    if (test_total % 5000 == 0) {
      LOG(INFO) << "Test: processed " << test_total << " lines, parsed " << test_parsed << " examples";
    }
  }
  
  LOG(INFO) << "=== Standard Split Data Summary ===";
  LOG(INFO) << "Training - Total: " << train_total << ", Parsed: " << train_parsed << ", Skipped: " << train_skipped;
  LOG(INFO) << "Test - Total: " << test_total << ", Parsed: " << test_parsed << ", Skipped: " << test_skipped;
  LOG(INFO) << "Train success rate: " << (100.0 * train_parsed / train_total) << "%";
  LOG(INFO) << "Test success rate: " << (100.0 * test_parsed / test_total) << "%";
  
  // 设置训练样本权重
  const float initial_wgt = 1.0 / train_examples->size();
  for (Example& example : *train_examples) {
    example.weight = initial_wgt;
  }
  
  // 输出特征和标签统计（修正后的代码）
  if (!train_examples->empty()) {
    LOG(INFO) << "Features per example: " << (*train_examples)[0].values.size();
  }
  
  LOG(INFO) << "===================================";
}