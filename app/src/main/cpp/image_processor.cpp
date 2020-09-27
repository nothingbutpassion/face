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
    Size textSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    int x0 = (left + right - textSize.width)/2;
    int y0 = top;
    rectangle(frame, Point(x0, y0), Point(x0 + textSize.width, y0-textSize.height-baseLine), color, FILLED);
    putText(frame, label, Point(x0, y0-baseLine), FONT_HERSHEY_SIMPLEX, 0.5, Scalar());
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
    if (!mCallingClassifier.load(modelDir))
        return false;
    LOGD("All model file is loaded from %s", modelDir.c_str());
    return true;
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

        vector<float> smoking_results;
        Mat smoking_image = mSmokingClassifier.getInput(gray, landmarks);
        mSmokingClassifier.predict(smoking_image, smoking_results);

        vector<float> left_calling_results;
        vector<float> right_calling_results;
        Mat left_calling_input = mCallingClassifier.getInput(gray, landmarks, true);
        Mat right_calling_input = mCallingClassifier.getInput(gray, landmarks, false);
        mCallingClassifier.predict(left_calling_input, left_calling_results);
        mCallingClassifier.predict(right_calling_input, right_calling_results);

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
        if (smoking_results[0] > 0.9)
            color = smoking;
        else if (left_calling_results[0] > 0.9 || right_calling_results[0] > 0.9)
            color = calling;
        string label = format("Smoke: %.2f, %.2f Call: %.2f, %.2f",
                smoking_results[0], smoking_results[1], left_calling_results[0], right_calling_results[0]);
        drawBoxes(image, box.x, box.y, box.x + box.width, box.y + box.height, label, color);

        // Draw landmarks
        drawLandmarks(image, landmarks);
    }

}

