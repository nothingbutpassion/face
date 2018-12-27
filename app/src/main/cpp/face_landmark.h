#ifndef FACE_FACE_LANDMARK_H
#define FACE_FACE_LANDMARK_H

#include <string>
#include <vector>
#include <opencv2/face.hpp>

class FaceLandmark {
public:
    FaceLandmark();
    bool load(const std::string& modelFile);
    bool fit(const cv::Mat& image, const std::vector<cv::Rect>& faces, std::vector<std::vector<cv::Point2f>>& landmarks);

private:
    cv::Ptr<cv::face::Facemark> mFacemark;
};


#endif
