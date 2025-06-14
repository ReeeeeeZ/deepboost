#include <glog/logging.h>

int main(int argc, char* argv[]) {
    google::InitGoogleLogging(argv[0]);
    // 设置日志输出目录
    google::SetLogDestination(google::INFO, "/mnt/e/25springcourse/MachineLearning/deepboost/glogtest/");
    
    LOG(INFO) << "Test message!";
    return 0;
}
