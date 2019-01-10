#ifndef FACE_FACE_DETECTOR_H
#define FACE_FACE_DETECTOR_H

#include <string>
#include <vector>
#include <opencv2/dnn.hpp>
#include "face_landmark.h"

class FaceDetector {
public:
    bool load(const std::string& modelDir);
    void detect(const cv::Mat& image, std::vector<cv::Rect>& objects);
    bool fit(const cv::Mat& image, const cv::Rect& face, std::vector<cv::Point2f>& landmarks);

    // Process all face-related stuff
    void process(cv::Mat& image);
private:
    FaceLandmark mFaceLandmark;
    cv::dnn::Net mFaceNet;

};


#endif //FACE_FACE_DETECTOR_H
