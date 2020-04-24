#ifndef KAZEMI_FACE_LANDMARK_H
#define KAZEMI_FACE_LANDMARK_H

#include <string>
#include <vector>
#include <opencv2/core.hpp>

class KazemiFaceLandmark {
public:
    ~KazemiFaceLandmark();
    bool load(const std::string& modelDir);
    void fit(const cv::Mat& gray, const cv::Rect& face, std::vector<cv::Point2f>& landmarks);
private:
    void* mShapePredictor = nullptr;

};

#endif // KAZEMI_FACE_LANDMARK_H
