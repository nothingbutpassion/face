#ifndef HOG_FACE_DETECTOR_H
#define HOG_FACE_DETECTOR_H

#include <string>
#include <vector>
#include <dlib/image_processing/frontal_face_detector.h>

class HOGFaceDetector {
public:
    bool load(const std::string& modelDir);
    void detect(const cv::Mat& image, std::vector<cv::Rect>& objects, std::vector<float>& confidences);

private:
    dlib::frontal_face_detector mFrontalFaceDetector;
};

#endif // HOG_FACE_DETECTOR_H