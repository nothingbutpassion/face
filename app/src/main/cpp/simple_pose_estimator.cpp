#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include "simple_pose_estimator.h"
#include "utils.h"

using namespace std;
using namespace cv;

#define LOG_TAG "SimplePoseEstimator"

// NOTES:
// 14 3D object points(world coordinates), the 3D head model comes from http://aifi.isr.uc.pt/Downloads/OpenGL/glAnthropometric3DModel.cpp
// Also see https://github.com/lincolnhard/head-pose-estimation
static vector<Point3d> objectPoints = {
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

static void adjustObjectPoints(vector<Point3d>& objectPoints, const vector<Point2d>& imagePoints,
        const Mat& cameraMatrix, const Mat& distCoeffs, const Mat& rvec, const Mat& tvec) {
    std::vector<Point3d> grads = objectPointGrads(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec);
    vector<Point3d> points = objectPoints;
    constexpr double lr = 1e-2;
    for (int i=0; i < points.size(); ++i) {
        points[i] -= lr*grads[i];
    }
    vector<double> oldErrors = projectErrors(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec);
    vector<double> newErrors = projectErrors(points, imagePoints, cameraMatrix, distCoeffs, rvec, tvec);
    double oldErr = 0;
    double newErr = 0;
    for (int i=0; i < points.size(); ++i) {
        oldErr += oldErrors[i];
        newErr += newErrors[i];
    }
    if (newErr < oldErr)
        objectPoints = points;
    LOGD("reproject errors: old=%.2f, new=%.2f", oldErr/points.size(), newErr/points.size());
}

//def reproject_axis(image, rotation_vec, translation_vec, cam_matrix, dist_coeffs):
//    orgsrc1 = (object_pts[4] + object_pts[5])/2 + np.float32([0, 0, 2])
//    reprojectsrc1 = 0.5*axis_pts + orgsrc1
//    orgsrc2 = (object_pts[6] + object_pts[7])/2 + np.float32([0, 0, 2])
//    reprojectsrc2 = 0.5*axis_pts + orgsrc2
//    # reprojectsrc =  np.concatenate((reprojectsrc1, reprojectsrc2))
//    reprojectsrc = (reprojectsrc1 + reprojectsrc2)/2
//    reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix, dist_coeffs)
//    reprojectdst = tuple(map(tuple, reprojectdst.reshape(-1, 2)))
//    cv2.line(image, reprojectdst[0], reprojectdst[1], (0, 0, 255), 2, cv2.LINE_AA)
//    cv2.line(image, reprojectdst[0], reprojectdst[2], (0, 255, 0), 2, cv2.LINE_AA)
//    cv2.line(image, reprojectdst[0], reprojectdst[3], (255, 0, 0), 2, cv2.LINE_AA)

static void drawAxis(Mat& image, const Mat& cameraMatrix, const Mat& distCoeffs, const Mat& rvec, const Mat& tvec) {
    std::vector<Point3d> axisPoints = {
            Point3d(0.0, 0.0, 0.0),
            Point3d(5.0, 0.0, 0.0),
            Point3d(0.0, 5.0, 0.0),
            Point3d(0.0, 0.0, 5.0)
    };
    Point3d t = (objectPoints[4] +  objectPoints[5] +  objectPoints[6] +  objectPoints[7])/4 + Point3d(0.0, 0.0, 2.0);
    for (Point3d& p: axisPoints) {
        p += t;
    }
    vector<Point2d> outPoints;
    projectPoints(axisPoints, rvec, tvec, cameraMatrix, distCoeffs, outPoints);
    line(image, outPoints[0], outPoints[1], CV_RGB(255, 0, 0), 2, LINE_AA);
    line(image, outPoints[0], outPoints[2], CV_RGB(0, 255, 0), 2, LINE_AA);
    line(image, outPoints[0], outPoints[3], CV_RGB(0, 0, 255), 2, LINE_AA);
}


bool SimplePoseEstimator::estimate(const Mat& image, const vector<Point2f>& faceMarks, vector<Point3f>& poses) {
    // NOTES:
    // There're 14 2D points from 68 face landmarks, see https://ibug.doc.ic.ac.uk/resources/300-W/
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
    // We apprximate cx by half image width, cy by half image height, fx = cx/tan(60/2*PI/180), fy=fx
    double cx = image.cols/2;
    double cy = image.rows/2;
    double fx = cx/tan(3.1415926535/6);
    double fy = fx;
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

    // Adjust object point coords by using "gradient descent" optimization
    adjustObjectPoints(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec);

    // Only for Debuging
    drawAxis(const_cast<Mat&>(image), cameraMatrix, distCoeffs, rvec, tvec);

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
    poses.push_back(Point3f(outEulerAngles.at<double>(0), outEulerAngles.at<double>(1), outEulerAngles.at<double>(2)));
    return true;
}

