#ifndef SIMPLE_POSE_ESTIMATOR_H
#define SIMPLE_POSE_ESTIMATOR_H

#include <vector>
#include <opencv2/core/types.hpp>
#include <opencv2/core.hpp>

class SimplePoseEstimator {
public:
    bool estimate(const cv::Mat& image, const std::vector<cv::Point2f>& faceMarks, std::vector<cv::Point3f>& poses);
};

#endif // SIMPLE_POSE_ESTIMATOR_H
