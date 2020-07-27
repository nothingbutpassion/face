#include <unistd.h>
#include <cmath>
#include <vector>
#include <tuple>
#include <sstream>
#include <thread>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <unistd.h>
#include "image_processor.h"
#include "resnet_face_descriptor.h"
#include "utils.h"

#define LOG_TAG "ImageProcessor"

using namespace std;
using namespace cv;

static void drawBoxes(const Mat& frame, int left, int top, int right, int bottom, const string& label, const Scalar& color) {
    rectangle(frame, Point(left, top), Point(right, bottom), color);
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - labelSize.height), Point(left + labelSize.width, top + baseLine), color, FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar());
}

static void drawLandmarks(Mat& image, const vector<Point2f>& landmarks) {
//    vector<Scalar> colors(68);
//    for (int i=0; i <= 16; ++i) colors[i] =  CV_RGB(255,0,0);  // 0 - 16 is profile      17 points
//    for (int i=17; i <= 21; ++i) colors[i] = CV_RGB(255,0,0);  // 17 - 21 left eyebrow    5 points
//    for (int i=22; i <= 26; ++i) colors[i] = CV_RGB(255,0,0);  // 22 - 26 right eyebrow   5 points
//    for (int i=27; i <= 30; ++i) colors[i] = CV_RGB(255,0,0);  // 27 - 30 nose bridge     4 points
//    for (int i=31; i <= 35; ++i) colors[i] = CV_RGB(255,0,0);  // 31 - 35 nose hole       5 points
//    for (int i=36; i <= 41; ++i) colors[i] = CV_RGB(0,0,255);  // 36 - 41 left eye        6 points
//    for (int i=42; i <= 47; ++i) colors[i] = CV_RGB(0,0,255);  // 42 - 47 right eye       6 points
//    for (int i=48; i <= 67; ++i) colors[i] = CV_RGB(255,0,0);  // 48 - 67 mouth           20 points
    for (int i = 0; i < landmarks.size(); ++i) {
        circle(image, landmarks[i], 2, CV_RGB(255, 0, 0), -1);
        //circle(image, landmarks[i], 2, colors[i], -1);
    }
}

static void drawPoses(Mat& image, const Point3f& eulerAngle) {
    stringstream outtext;
    outtext << "x-rotation: " << std::setprecision(3) << eulerAngle.x;
    cv::putText(image, outtext.str(), cv::Point(4, 96), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(255, 0, 0));
    outtext.str("");
    outtext << "y-rotation: " << std::setprecision(3) << eulerAngle.y;
    cv::putText(image, outtext.str(), cv::Point(4, 120), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 0));
    outtext.str("");
    outtext << "z-rotation: " << std::setprecision(3) << eulerAngle.z;
    cv::putText(image, outtext.str(), cv::Point(4, 144), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 255));
}

static void drawAxis(Mat& image, const vector<Point3d>& objectPoints, const Mat& cameraMatrix, const Mat& distCoeffs, const Mat& rvec, const Mat& tvec) {
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



ImageProcessor::ImageProcessor() {}
ImageProcessor::~ImageProcessor() {}

bool ImageProcessor::init(const string& modelDir) {
    if (!mFaceDetector.load(modelDir))
        return false;
    if (!mFaceLandmark.load(modelDir))
        return false;
    if (!mSmokingClassifier.load(modelDir))
        return false;
    LOGD("All model file is loaded from %s", modelDir.c_str());
    return true;
}

Rect correct_box(const Size& imgSize, const vector<Point2f>& landmarks) {
    int x_min = landmarks[1].x;
    int x_max = landmarks[15].x;
    int y_min = landmarks[19].y;
    int y_max = landmarks[8].y;
    y_min = y_min - (y_max - y_min) / 6;
    Rect r;
    r.x = max(x_min, 0);
    r.y = max(y_min, 0);
    r.width = min(x_max-x_min+1, imgSize.width - r.x);
    r.height = min(y_max-y_min+1, imgSize.height - r.y);
    return r;
}


Rect extend_box(const Size& imgSize, const Rect& faceBox) {
    Rect r = faceBox;
    int d = max(r.width, r.height)/3;
    r.x -= d;
    r.width += 2*d;
    if (r.width > r.height + 3*d/2)
        r.height += 3*d/2;
    else
        r.height = r.width;
    r.x = max(0, r.x);
    r.y = max(0, r.y);
    r.width = min(r.width, imgSize.width-r.x);
    r.height = min(r.height, imgSize.height-r.y);
    return r;
}

Mat get_smoking_image(const Mat& image, const vector<Point2f>& landmarks) {
    Point2f leftEye(0, 0);
    Point2f rightEye(0, 0);
    for (int i = 42; i < 48; ++i)
        leftEye += landmarks[i];
    for (int i = 36; i < 42; ++i)
        rightEye += landmarks[i];
    Point2f center((rightEye.x+leftEye.x)/12, (rightEye.y+leftEye.y)/12);
    float angle = atan2f(rightEye.y-leftEye.y, rightEye.x-leftEye.x)*180/3.1415926535 + 180;
    Mat M = getRotationMatrix2D(center, angle, 1.0);
    vector<Point2f> marks;
    float x_max = -1000000;
    float x_min = 1000000;
    for (const Point2f& p: landmarks) {
        float x = p.x*M.at<double>(0, 0) + p.y*M.at<double>(0, 1) + M.at<double>(0, 2);
        float y = p.x*M.at<double>(1, 0) + p.y*M.at<double>(1, 1) + M.at<double>(1, 2);
        x_max = max(x, x_max);
        x_min = min(x, x_min);
        marks.push_back(Point2f(x, y));
    }
    float x0 = marks[33].x;
    float y0 = marks[33].y;
    float w = 0.6*(x_max - x_min);
    float h = min(w*2/3,  image.cols - y0);
    M.at<double>(0,2) -= (x0 - 0.5*w);
    M.at<double>(1,2) -= y0;
    M.at<double>(0,0) *= 64/w;
    M.at<double>(0,1) *= 64/w;
    M.at<double>(0,2) *= 64/w;
    M.at<double>(1,0) *= 64/h;
    M.at<double>(1,1) *= 64/h;
    M.at<double>(1,2) *= 64/h;
    Mat dst;
    warpAffine(image, dst, M, Size(64, 64), INTER_CUBIC);
    return dst;
}

void ImageProcessor::process(Mat& image) {
    Mat gray;
    Mat gray_hog;
    vector<Rect> boxes;
    vector<float> scores;
    vector<int> indices;

    cvtColor(image, gray, COLOR_RGBA2GRAY);
    resize(gray, gray_hog, gray.size()/2);

    mFaceDetector.detect(gray_hog, boxes, scores, &indices);
    for (Rect& b: boxes) {
        b.x *= 2;
        b.y *= 2;
        b.width *= 2;
        b.height *= 2;
    }
    if (boxes.size() > 0) {
        Rect box = boxes[0];
        vector<Point2f> landmarks;
        mFaceLandmark.fit(gray, box, landmarks);

        Mat smoking_image = get_smoking_image(gray, landmarks);
        vector<float> actions;
        mSmokingClassifier.predict(smoking_image, actions);

        // Pose estimation
        Point3f pose;
        mPoseEstimator.estimate(image.size(), landmarks, pose);
        drawAxis(image, mPoseEstimator.objectPoints(), mPoseEstimator.cameraMatrix(),
                mPoseEstimator.distCoeffs(), mPoseEstimator.rvec(), mPoseEstimator.tvec());

        // Draw face boxes
        Scalar smoking = cv::Scalar(255, 0, 0, 0);
        Scalar calling = cv::Scalar(255, 255, 0, 0);
        Scalar normal = cv::Scalar(0, 255, 0, 0);
        Scalar color = normal;
        if (actions[0] > 0.8)
            color = smoking;
        else if (actions[1] > 0.8)
            color = calling;
        string label = format("action: %.2f, %.2f, %.2f", actions[0], actions[1], actions[2]);
        drawBoxes(image, box.x, box.y, box.x + box.width, box.y + box.height, label, color);

        // Draw landmarks
        drawLandmarks(image, landmarks);
    }

}

