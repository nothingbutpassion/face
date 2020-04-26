#ifndef ACTION_CLASSIFIER_H
#define ACTION_CLASSIFIER_H

#include <string>
#include <vector>
#include <opencv2/dnn.hpp>

class ActionClassifier {
public:
    bool load(const std::string& modelDir);
    void predict(const cv::Mat& image, std::vector<float>& prediction);

private:
    cv::dnn::Net mClassifierNet;
};


#endif // ACTION_CLASSIFIER_H
