#ifndef KAZEMI_FACE_LANDMARK_H
#define KAZEMI_FACE_LANDMARK_H

#include <string>
#include <vector>
//#include <opencv2/face.hpp>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>

class KazemiFaceLandmark {
public:
    bool load(const std::string& modelDir);
    bool fit(const cv::Mat& image, const std::vector<cv::Rect>& faces, std::vector<std::vector<cv::Point2f>>& landmarks);

private:
    dlib::shape_predictor mShapePredictor;
};

#endif // KAZEMI_FACE_LANDMARK_H
