#ifndef CAFFE_ACTION_CLASSIFIER_H
#define CAFFE_ACTION_CLASSIFIER_H

#include <string>
#include <vector>
#include <opencv2/dnn.hpp>

class CaffeActionClassifier {
public:
    bool load(const std::string& modelDir);
    void predict(const cv::Mat& image, std::vector<float>& results, const cv::Scalar& meanBGR=cv::Scalar(76, 89, 103));

private:
    cv::dnn::Net mClassifierNet;
};


#endif  // CAFFE_ACTION_CLASSIFIER_H