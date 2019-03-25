#include <opencv2/imgproc.hpp>
#include <dlib/opencv.h>
#include "hog_face_detector.h"


bool HOGFaceDetector::load(const std::string& modelDir) {
    //
    //  No need to load model from file
    //
    mFrontalFaceDetector = dlib::get_frontal_face_detector();
    return true;
}
void HOGFaceDetector::detect(const cv::Mat& image, std::vector<cv::Rect>& objects, std::vector<float>& confidences) {
    cv::Mat small;
    cv::Mat gray;
    cv::resize(image, small, image.size()/2);
    cv::cvtColor(small, gray, cv::COLOR_RGBA2GRAY);
    dlib::cv_image<uchar > cimg(gray);
    // NOTES:
    // cv::rectangle is also existed
    std::vector<std::pair<double, dlib::rectangle>> faces;
    mFrontalFaceDetector(cimg, faces);
    // Find the pose of each face.
    for (auto& face: faces) {
        objects.push_back(cv::Rect(face.second.left()*2, face.second.top()*2, face.second.width()*2, face.second.height()*2));
        confidences.push_back(face.first);
    }
}

