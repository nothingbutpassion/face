#ifndef CAFFE_SMOKING_CLASSIFIER_H
#define CAFFE_SMOKING_CLASSIFIER_H

#include <string>
#include <vector>
#include <opencv2/dnn.hpp>

class CaffeSmokingClassifier {
public:
    bool load(const std::string& modelDir);
    void predict(const cv::Mat& gray, std::vector<float>& results);
    static cv::Mat getInput(const cv::Mat& gray, const std::vector<cv::Point2f>& landmarks);
private:
    cv::dnn::Net mClassifierNet;
};

#endif // CAFFE_SMOKING_CLASSIFIER_H
