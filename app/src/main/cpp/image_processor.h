#ifndef DMSWORKER_H
#define DMSWORKER_H

#include <string>
#include <opencv2/core.hpp>

#include "hog_face_detector.h"
#include "kazemi_face_landmark.h"
#include "simple_pose_estimator.h"
#include "resnet_face_descriptor.h"
#include "action_classifier.h"

class ImageProcessor {
public:
    ImageProcessor();
    ~ImageProcessor();
    bool init(const std::string& modelDir);
    void process(cv::Mat& image);

private:
    HOGFaceDetector mFaceDetector;
    KazemiFaceLandmark mFaceLandmark;
    SimplePoseEstimator mPoseEstimator;
    ActionClassifier mActionClassifier;
};


#endif // DMSWORKER_H
