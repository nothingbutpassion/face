#ifndef CAFFE_CALLING_CLASSIFIER_H
#define CAFFE_CALLING_CLASSIFIER_H

#include <string>
#include <vector>
#include <opencv2/dnn.hpp>

class CaffeCallingClassifier {
public:
    bool load(const std::string& modelDir);
    void predict(const cv::Mat& gray, std::vector<float>& results);
    static cv::Mat getInput(const cv::Mat& gray, const std::vector<cv::Point2f>& landmarks);
private:
    cv::dnn::Net mClassifierNet;
};

#endif // CAFFE_CALLING_CLASSIFIER_H
