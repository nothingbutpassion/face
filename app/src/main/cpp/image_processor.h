#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

#include <string>
#include <opencv2/core.hpp>

#include "hog_face_detector.h"
#include "kazemi_face_landmark.h"
#include "simple_pose_estimator.h"
#include "caffe_smoking_classifier.h"
#include "caffe_calling_classifier.h"

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
    CaffeSmokingClassifier mSmokingClassifier;
    CaffeCallingClassifier mCallingClassifier;
};


#endif // IMAGE_PROCESSOR_H
