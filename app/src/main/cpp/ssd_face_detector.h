#ifndef FACE_DETECTOR_H
#define FACE_DETECTOR_H

#include <string>
#include <vector>
#include <opencv2/dnn.hpp>

class SSDFaceDetector {
public:
    bool load(const std::string& modelDir);
    void detect(const cv::Mat& image, std::vector<cv::Rect>& objects, std::vector<float>& confidences);
private:
    cv::dnn::Net mFaceNet;
};


#endif // FACE_DETECTOR_H
