#ifndef DMSWORKER_H
#define DMSWORKER_H

#include <string>
#include <opencv2/core.hpp>

#include "hao_face_detector.h"
#include "kazemi_face_landmark.h"

class ImageProcessor {
public:
    bool init(const std::string& modelDir);
    void process(cv::Mat& image);

private:
    HaoFaceDetector mFaceDetector;
    KazemiFaceLandmark mFaceLandmark;
};


#endif // DMSWORKER_H
