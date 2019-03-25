#ifndef DMSWORKER_H
#define DMSWORKER_H

#include <string>
#include <opencv2/core.hpp>

#include "hog_face_detector.h"
#include "lbf_face_landmark.h"

class DMSWorker {
public:
    bool init(const std::string& modelDir);
    void process(cv::Mat& image);

private:
    HOGFaceDetector mFaceDetector;
    LBFFaceLandmark mFaceLandmark;
};


#endif // DMSWORKER_H
