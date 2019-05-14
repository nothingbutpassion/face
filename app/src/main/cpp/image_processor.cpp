#import <vector>
#include <sstream>
#import <opencv2/opencv.hpp>
#include "image_processor.h"

using namespace std;
using namespace cv;

static void drawBoxes(const Mat& frame, float confidence, int left, int top, int right, int bottom) {
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0));
    string label = format("%.2f", confidence);

    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - labelSize.height), Point(left + labelSize.width, top + baseLine), Scalar::all(255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar());
}

static void drawLandmarks(Mat& image, const vector<Rect>& boxes, const vector<vector<Point2f>>& landmarks) {
    for (int i=0; i < boxes.size(); ++i){
        for (int j = 0; j < landmarks[i].size(); ++j) {
            circle(image, landmarks[i][j], 2, CV_RGB(255, 0, 0), -1);
        }
    }
}

static void drawPoses(Mat& image, const vector<Point3f>& poses) {
    Point3f eulerAngle = poses[0];
    stringstream outtext;
    outtext << "X: " << std::setprecision(3) << eulerAngle.x;
    cv::putText(image, outtext.str(), cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(255, 0, 0));
    outtext.str("");
    outtext << "Y: " << std::setprecision(3) << eulerAngle.y;
    cv::putText(image, outtext.str(), cv::Point(20, 60), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 0));
    outtext.str("");
    outtext << "Z: " << std::setprecision(3) << eulerAngle.z;
    cv::putText(image, outtext.str(), cv::Point(20, 80), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 255));
    outtext.str("");
}

bool ImageProcessor::init(const string& modelDir) {
    if (!mFaceDetector.load(modelDir)) {
        return false;
    }
    if (!mFaceLandmark.load(modelDir)) {
        return false;
    }
    return true;
}


void ImageProcessor::process(Mat& image) {
    vector<Rect> boxes;
    vector<float> confidences;
    mFaceDetector.detect(image, boxes, confidences);
    if (boxes.size() > 0) {
        for (int i=0; i < boxes.size(); ++i) {
            // Draw face boxes
            drawBoxes(image, confidences[i], boxes[i].x, boxes[i].y, boxes[i].x + boxes[i].width, boxes[i].y + boxes[i].height);
        }
        vector<vector<Point2f>> landmarks;
        mFaceLandmark.fit(image, boxes, landmarks);
        // Draw landmarks
        drawLandmarks(image, boxes, landmarks);

        vector<Point3f> poses;
        mPoseEstimator.estimate(image, landmarks[0], poses);
        // Draw head pose
        drawPoses(image, poses);
    }

}

