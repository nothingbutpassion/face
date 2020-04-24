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
    cv::Mat extract(const cv::Mat& image, const cv::Rect& face, std::vector<cv::Point2f>& landmark, cv::Mat* chip= nullptr);
    float distance(const cv::Mat& descriptor1, const cv::Mat& descriptor2);


private:
    void* mResNet = nullptr;
};


#endif //RESNET_FACE_DESCRIPTOR_H
