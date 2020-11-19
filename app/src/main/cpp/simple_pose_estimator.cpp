#include <opencv2/calib3d/calib3d.hpp>
#include "simple_pose_estimator.h"
#include "utils.h"

using namespace std;
using namespace cv;

#define LOG_TAG "SimplePoseEstimator"


// NOTES:
// 14 3D object points(world coordinates), the 3D head model comes from http://aifi.isr.uc.pt/Downloads/OpenGL/glAnthropometric3DModel.cpp
// Also see https://github.com/lincolnhard/head-pose-estimation
/*
static vector<Point3d> kReferencePoints = {
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
*/
static vector<Point3d> kReferencePoints = {
        Point3d(-6.825897, 6.760612, 4.402142),     //#33 left brow left corner
        Point3d(-1.330353, 7.122144, 6.903745),     //#29 left brow right corner
        Point3d( 1.330353, 7.122144, 6.903745),     //#34 right brow left corner
        Point3d( 6.825897, 6.760612, 4.402142),     //#38 right brow right corner
        Point3d(-5.311432, 5.485328, 3.987654),     //#13 left eye left corner
        Point3d(-1.789930, 5.393625, 4.413414),     //#17 left eye right corner
        Point3d( 1.789930, 5.393625, 4.413414),     //#25 right eye left corner
        Point3d( 5.311432, 5.485328, 3.987654),     //#21 right eye right corner
        Point3d(-2.005628, 1.409845, 6.165652),     //#55 nose left corner
        Point3d( 2.005628, 1.409845, 6.165652),     //#49 nose right corner
        Point3d(-2.774015, -2.080775, 5.048531),    //#43 mouth left corner
        Point3d( 2.774015, -2.080775, 5.048531),    //#39 mouth right corner
        Point3d( 0.000000, -3.116408, 6.097667),    //#45 mouth central bottom corner
        Point3d( 0.000000, -7.415691, 4.070434)     //#6 chin corner
};


static vector<double> projectErrors(const vector<Point3d>& objectPoints, const vector<Point2d>& imagePoints,
        const Mat& cameraMatrix, const Mat& distCoeffs, const Mat& rvec, const Mat& tvec) {
    vector<double> errors(objectPoints.size(), 0);
    vector<Point2d> outPoints;
    projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs, outPoints);
    for (int i=0; i < imagePoints.size(); ++i) {
        Point2d d = imagePoints[i] - outPoints[i];
        errors[i] = sqrt(d.x*d.x + d.y*d.y);
    }
    return errors;
}

static std::vector<Point3d> objectPointGrads(const vector<Point3d>& objectPoints, const vector<Point2d>& imagePoints,
        const Mat& cameraMatrix, const Mat& distCoeffs, const Mat& rvec, const Mat& tvec) {
    vector<double> f = projectErrors(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec);
    vector<Point3d> px = objectPoints;
    vector<Point3d> py = objectPoints;
    vector<Point3d> pz = objectPoints;
    constexpr double delta = 1e-4;
    for (int i=0; i < objectPoints.size(); ++i) {
        px[i].x += delta;
        py[i].y += delta;
        pz[i].z += delta;
    }
    vector<double> fx = projectErrors(px, imagePoints, cameraMatrix, distCoeffs, rvec, tvec);
    vector<double> fy = projectErrors(py, imagePoints, cameraMatrix, distCoeffs, rvec, tvec);
    vector<double> fz = projectErrors(pz, imagePoints, cameraMatrix, distCoeffs, rvec, tvec);
    std::vector<Point3d> grads;
    for (int i=0; i < objectPoints.size(); ++i) {
        grads.push_back(Point3d((fx[i]-f[i])/delta, (fy[i]-f[i])/delta, (fz[i]-f[i])/delta));
    }
    return grads;
}

//static void adjustObjectPoints(vector<Point3d>& objectPoints, const vector<Point2d>& imagePoints,
//        const Mat& cameraMatrix, const Mat& distCoeffs, const Mat& rvec, const Mat& tvec) {
//    std::vector<Point3d> grads = objectPointGrads(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec);
//    vector<Point3d> points = objectPoints;
//    constexpr double lr = 1e-2;
//    constexpr double thresh = 4;
//    for (int i=0; i < points.size(); ++i) {
//        Point3d d = points[i]- lr*grads[i] - kReferencePoints[i];
//        if (d.x*d.x + d.y*d.y + d.z*d.z < thresh)
//            points[i] -= lr*grads[i];
//    }
//    vector<double> refErrors = projectErrors(kReferencePoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec);
//    vector<double> curErrors = projectErrors(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec);
//    vector<double> newErrors = projectErrors(points, imagePoints, cameraMatrix, distCoeffs, rvec, tvec);
//    double refErr = 0;
//    double curErr = 0;
//    double newErr = 0;
//    for (int i=0; i < points.size(); ++i) {
//        refErr += refErrors[i];
//        curErr += curErrors[i];
//        newErr += newErrors[i];
//    }
//    if (newErr < curErr && newErr < refErr)
//        objectPoints = points;
//    else if (refErr < curErr)
//        objectPoints = kReferencePoints;
//    //LOGD("project errors: reference=%.2f, current=%.2f, adjusted=%.2f",
//    //        refErr/points.size(), curErr/points.size(), newErr/points.size());
//}

static void adjustObjectPoints(vector<Point3d>& objectPoints, const vector<Point2d>& imagePoints,
        const Mat& cameraMatrix, const Mat& distCoeffs, const Mat& rvec, const Mat& tvec) {
    std::vector<Point3d> grads = objectPointGrads(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec);
    vector<Point3d> points = objectPoints;
    constexpr double lr = 1e-2;
    constexpr double thresh = 4;
    for (int i=0; i < points.size(); ++i) {
        Point3d d = points[i]- lr*grads[i] - kReferencePoints[i];
        if (d.x*d.x + d.y*d.y + d.z*d.z < thresh)
            points[i] -= lr*grads[i];
    }
    vector<double> refErrors = projectErrors(kReferencePoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec);
    vector<double> curErrors = projectErrors(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec);
    vector<double> newErrors = projectErrors(points, imagePoints, cameraMatrix, distCoeffs, rvec, tvec);
    double refErr = 0;
    double curErr = 0;
    double newErr = 0;
    for (int i=0; i < points.size(); ++i) {
        if (newErrors[i] < curErrors[i] && newErrors[i] < refErrors[i])
            objectPoints[i] = points[i];
        else if (refErrors[i] < curErrors[i])
            objectPoints[i] = kReferencePoints[i];
        refErr += refErrors[i];
        curErr += curErrors[i];
        newErr += newErrors[i];
    }
    LOGI("project errors: reference=%.2f, current=%.2f, adjusted=%.2f",
            refErr/points.size(), curErr/points.size(), newErr/points.size());
}

SimplePoseEstimator::SimplePoseEstimator() {
    mObjectPoints = kReferencePoints;
}

void SimplePoseEstimator::project(const vector<Point3d>& objectPoints, const Mat& rvec,
        const Mat& tvec, const vector<Point3d>& imagePoints ) {
    projectPoints(objectPoints, rvec, tvec, mCameraMatrix, mDistCoeffs, imagePoints);
}

bool SimplePoseEstimator::estimate(const Size& size, const vector<Point2f>& landmarks,
        Point3f& eulerAngle, Mat* rotation, cv::Mat* translation, bool adjust) {
    // NOTES:
    // We approximate Camera Matrix  with cx = half image width, cy = half image height, fx = cx/tan(60/2*PI/180), fy=fx
    if (mCameraMatrix.empty()) {
        double cx = size.width/2;
        double cy = size.height/2;

        // NOTES:
        // For hi3559av100 camera sensor, the FOV is about 30 degrees (cx/fx == 33/140)
        // double fx = cx/tan(3.1415926535/6);
        double fx = cx*4.24;
        double fy = fx;
        double K[9] = {
                fx, 0.0, cx,
                0.0, fy, cy,
                0.0, 0.0, 1.0
        };
        double D[5] = { 0.0, 0.0, 0.0, 0.0, 0.0 };
        Mat(3, 3, CV_64FC1, K).copyTo(mCameraMatrix);
        Mat(5, 1, CV_64FC1, D).copyTo(mDistCoeffs);
    }
    // NOTES:
    // There're 14 2D points from 68 face landmarks, see https://ibug.doc.ic.ac.uk/resources/300-W/
    std::vector<Point2d> imagePoints = {
            cv::Point2d(landmarks[17].x, landmarks[17].y), //#17 left brow left corner
            cv::Point2d(landmarks[21].x, landmarks[21].y), //#21 left brow right corner
            cv::Point2d(landmarks[22].x, landmarks[22].y), //#22 right brow left corner
            cv::Point2d(landmarks[26].x, landmarks[26].y), //#26 right brow right corner
            cv::Point2d(landmarks[36].x, landmarks[36].y), //#36 left eye left corner
            cv::Point2d(landmarks[39].x, landmarks[39].y), //#39 left eye right corner
            cv::Point2d(landmarks[42].x, landmarks[42].y), //#42 right eye left corner
            cv::Point2d(landmarks[45].x, landmarks[45].y), //#45 right eye right corner
            cv::Point2d(landmarks[31].x, landmarks[31].y), //#31 nose left corner
            cv::Point2d(landmarks[35].x, landmarks[35].y), //#35 nose right corner
            cv::Point2d(landmarks[48].x, landmarks[48].y), //#48 mouth left corner
            cv::Point2d(landmarks[54].x, landmarks[54].y), //#54 mouth right corner
            cv::Point2d(landmarks[57].x, landmarks[57].y), //#57 mouth central bottom corner
            cv::Point2d(landmarks[8].x,  landmarks[8].y),  //#8 chin corner
    };

    // Get Rotation and Translation vector
    solvePnP(mObjectPoints, imagePoints, mCameraMatrix, mDistCoeffs, mRVec, mTVec);

    // Adjust object point coords by using "gradient descent" optimization
    if (adjust)
        adjustObjectPoints(mObjectPoints, imagePoints, mCameraMatrix, mDistCoeffs, mRVec, mTVec);

    // Caculate Euler angles (Unit: degrees)
    Mat rotMatrix(3, 3, CV_64FC1);
    Mat projMatrix(3, 4, CV_64FC1);
    Mat outCamMatrix(3, 3, CV_64FC1);
    Mat outRotMatrix(3, 3, CV_64FC1);
    Mat outTransVec(4, 1, CV_64FC1);
    Mat outEulerAngles(3, 1, CV_64FC1);
    cv::Rodrigues(mRVec, rotMatrix);
    cv::hconcat(rotMatrix, mTVec, projMatrix);
    decomposeProjectionMatrix(projMatrix, outCamMatrix, outRotMatrix, outTransVec, cv::noArray(), cv::noArray(), cv::noArray(), outEulerAngles);
    eulerAngle = Point3f(outEulerAngles.at<double>(0), outEulerAngles.at<double>(1), outEulerAngles.at<double>(2));
    if (rotation)
        *rotation = mRVec;
    if (translation)
        *translation = mTVec;
    return true;
}



