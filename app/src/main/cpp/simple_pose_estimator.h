#ifndef SIMPLE_POSE_ESTIMATOR_H
#define SIMPLE_POSE_ESTIMATOR_H

#include <vector>
#include <opencv2/core.hpp>

class SimplePoseEstimator {
public:
    SimplePoseEstimator();
    bool estimate(const cv::Size& size, const std::vector<cv::Point2f>& landmarks,
            std::vector<cv::Point3f>& aulerAngles, cv::Mat* rvec = nullptr, cv::Mat* tvec = nullptr);
    void project(const std::vector<cv::Point3d>& objectPoints, const cv::Mat& rvec,
            const cv::Mat& tvec, const std::vector<cv::Point3d>& imagePoints);
    const std::vector<cv::Point3d>& objectPoints() const { return mObjectPoints; }
    const cv::Mat& cameraMatrix() const { return mCameraMatrix; }
    const cv::Mat& distCoeffs() const { return mDistCoeffs; }
private:
    std::vector<cv::Point3d> mObjectPoints;
    cv::Mat mCameraMatrix;
    cv::Mat mDistCoeffs;
};

#endif // SIMPLE_POSE_ESTIMATOR_H
