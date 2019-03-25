#import <vector>
#import <opencv2/opencv.hpp>
#include "dmsworker.h"

using namespace std;
using namespace cv;

static void drawDetection(const Mat& frame, float confidence, int left, int top, int right, int bottom) {
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0));
    string label = format("%.2f", confidence);

    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - labelSize.height), Point(left + labelSize.width, top + baseLine), Scalar::all(255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar());
}

bool DMSWorker::init(const string& modelDir) {
    if (!mFaceDetector.load(modelDir)) {
        return false;
    }
    if (!mFaceLandmark.load(modelDir)) {
        return false;
    }
    return true;
}


void DMSWorker::process(Mat& image) {
    vector<Rect> boxes;
    vector<float> confidences;
    mFaceDetector.detect(image, boxes, confidences);
    if (boxes.size() > 0) {
        for (int i=0; i < boxes.size(); ++i) {
            drawDetection(image, confidences[i], boxes[i].x, boxes[i].y, boxes[i].x + boxes[i].width, boxes[i].y + boxes[i].height);
        }
        vector<vector<Point2f>> landmarks;
        mFaceLandmark.fit(image, boxes, landmarks);
    }

}

