#ifndef FACE_FACE_DETECTOR_H
#define FACE_FACE_DETECTOR_H

#include <string>
#include <vector>
#include <opencv2/objdetect.hpp>
#include <opencv2/face.hpp>
#include "face_landmark.h"

class FaceDetector {
public:
    bool load(const std::string& modelFile);
    void detect(const cv::Mat& image, std::vector<cv::Rect>& objects);
    bool fit(const cv::Mat& image, const cv::Rect& face, std::vector<cv::Point2f>& landmarks);
private:
    cv::CascadeClassifier mFaceClassifier;
    cv::CascadeClassifier mEyeClassifier;
    FaceLandmark mFaceLandmark;

};


#endif //FACE_FACE_DETECTOR_H
