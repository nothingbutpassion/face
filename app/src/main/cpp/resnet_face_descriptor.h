#ifndef RESNET_FACE_DESCRIPTOR_H
#define RESNET_FACE_DESCRIPTOR_H

#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <dlib/image_processing.h>

class ResnetFaceDescriptor {
public:
    ~ResnetFaceDescriptor();
    bool load(const std::string& modelDir);
    dlib::matrix<float,0,1> extract(const cv::Mat& image, const cv::Rect& face, const std::vector<cv::Point2f>& landmark);
    float distance(const dlib::matrix<float,0,1>& descriptor1, const dlib::matrix<float,0,1>& descriptor2);
private:
    void* mResNet = nullptr;
};


#endif //RESNET_FACE_DESCRIPTOR_H
