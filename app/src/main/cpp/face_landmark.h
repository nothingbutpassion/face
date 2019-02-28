#ifndef FACE_LANDMARK_H
#define FACE_LANDMARK_H

#include <string>
#include <vector>
#include <opencv2/face.hpp>

class FaceLandmark {
public:
    FaceLandmark();
    bool load(const std::string& modelDir);
    bool fit(const cv::Mat& image, const std::vector<cv::Rect>& faces, std::vector<std::vector<cv::Point2f>>& landmarks);

private:
    cv::Ptr<cv::face::Facemark> mFacemark;
};


#endif // FACE_LANDMARK_H
