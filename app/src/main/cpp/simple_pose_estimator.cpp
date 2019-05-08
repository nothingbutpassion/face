#include <opencv2/calib3d/calib3d.hpp>
#include "simple_pose_estimator.h"

using namespace std;
using namespace cv;

// NOTES:
// Object points(world coordinates), the 3D head model comes from http://aifi.isr.uc.pt/Downloads/OpenGL/glAnthropometric3DModel.cpp
// Also see https://github.com/lincolnhard/head-pose-estimation
static std::vector<Point3d> objectPoints = {
    Point3d(6.825897, 6.760612, 4.402142),     //#33 left brow left corner
    Point3d(1.330353, 7.122144, 6.903745),     //#29 left brow right corner
    Point3d(-1.330353, 7.122144, 6.903745),    //#34 right brow left corner
    Point3d(-6.825897, 6.760612, 4.402142),    //#38 right brow right corner
    Point3d(5.311432, 5.485328, 3.987654),     //#13 left eye left corner
    Point3d(1.789930, 5.393625, 4.413414),     //#17 left eye right corner
    Point3d(-1.789930, 5.393625, 4.413414),    //#25 right eye left corner
    Point3d(-5.311432, 5.485328, 3.987654),    //#21 right eye right corner
    Point3d(2.005628, 1.409845, 6.165652),     //#55 nose left corner
    Point3d(-2.005628, 1.409845, 6.165652),    //#49 nose right corner
    Point3d(2.774015, -2.080775, 5.048531),    //#43 mouth left corner
    Point3d(-2.774015, -2.080775, 5.048531),   //#39 mouth right corner
    Point3d(0.000000, -3.116408, 6.097667),    //#45 mouth central bottom corner
    Point3d(0.000000, -7.415691, 4.070434)     //#6 chin corner
};





bool SimplePoseEstimator::estimate(const Mat& image, const vector<Point2f>& faceMarks, vector<Point3f>& poses) {
    // NOTES:
    // Select image points from 68 faceMarks, see https://ibug.doc.ic.ac.uk/resources/300-W/
    std::vector<Point2d> imagePoints = {
            cv::Point2d(faceMarks[17].x, faceMarks[17].y), //#17 left brow left corner
            cv::Point2d(faceMarks[21].x, faceMarks[21].y), //#21 left brow right corner
            cv::Point2d(faceMarks[22].x, faceMarks[22].y), //#22 right brow left corner
            cv::Point2d(faceMarks[26].x, faceMarks[26].y), //#26 right brow right corner
            cv::Point2d(faceMarks[36].x, faceMarks[36].y), //#36 left eye left corner
            cv::Point2d(faceMarks[39].x, faceMarks[39].y), //#39 left eye right corner
            cv::Point2d(faceMarks[42].x, faceMarks[42].y), //#42 right eye left corner
            cv::Point2d(faceMarks[45].x, faceMarks[45].y), //#45 right eye right corner
            cv::Point2d(faceMarks[31].x, faceMarks[31].y), //#31 nose left corner
            cv::Point2d(faceMarks[35].x, faceMarks[35].y), //#35 nose right corner
            cv::Point2d(faceMarks[48].x, faceMarks[48].y), //#48 mouth left corner
            cv::Point2d(faceMarks[54].x, faceMarks[54].y), //#54 mouth right corner
            cv::Point2d(faceMarks[57].x, faceMarks[57].y), //#57 mouth central bottom corner
            cv::Point2d(faceMarks[8].x,  faceMarks[8].y),   //#8 chin corner
    };
    // NOTES:
    // We apprximate fx and fy by image width, cx by half image width, cy by half image height instead
    double fx = image.cols;
    double fy = image.cols;
    double cx = image.cols/2;
    double cy = image.rows/2;
    double K[9] = {
            fx, 0.0, cx,
            0.0, fy, cy,
            0.0, 0.0, 1.0
    };
    double D[5] = { 0.0, 0.0, 0.0, 0.0, 0.0 };
    Mat cameraMatrix(3, 3, CV_64FC1, K);
    Mat distCoeffs(5, 1, CV_64FC1, D);
    Mat rvec;
    Mat tvec;
    // Get Rotation and Translation vector
    solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec);

    // Caculate Euler angles (Unit: degrees)
    Mat rotMatrix(3, 3, CV_64FC1);
    Mat projMatrix(3, 4, CV_64FC1);
    Mat outCamMatrix(3, 3, CV_64FC1);
    Mat outRotMatrix(3, 3, CV_64FC1);
    Mat outTransVec(4, 1, CV_64FC1);
    Mat outEulerAngles(3, 1, CV_64FC1);
    cv::Rodrigues(rvec, rotMatrix);
    cv::hconcat(rotMatrix, tvec, projMatrix);
    decomposeProjectionMatrix(projMatrix, outCamMatrix, outRotMatrix, outTransVec, cv::noArray(), cv::noArray(), cv::noArray(), outEulerAngles);

    // Fill outputs.
    poses.push_back(Point3f(outEulerAngles.at<double>(2), outEulerAngles.at<double>(1), outEulerAngles.at<double>(2)));
    return true;
}

