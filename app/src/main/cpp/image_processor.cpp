#include <unistd.h>
#include <vector>
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
//    vector<Scalar> colors(68);
//    for (int i=0; i <= 16; ++i) colors[i] =  CV_RGB(255,0,0);  // 0 - 16 is profile      17 points
//    for (int i=17; i <= 21; ++i) colors[i] = CV_RGB(255,0,0);  // 17 - 21 left eyebrow    5 points
//    for (int i=22; i <= 26; ++i) colors[i] = CV_RGB(255,0,0);  // 22 - 26 right eyebrow   5 points
//    for (int i=27; i <= 30; ++i) colors[i] = CV_RGB(255,0,0);  // 27 - 30 nose bridge     4 points
//    for (int i=31; i <= 35; ++i) colors[i] = CV_RGB(255,0,0);  // 31 - 35 nose hole       5 points
//    for (int i=36; i <= 41; ++i) colors[i] = CV_RGB(0,0,255);    // 36 - 41 left eye        6 points
//    for (int i=42; i <= 47; ++i) colors[i] = CV_RGB(0,0,255);    // 42 - 47 right eye       6 points
//    for (int i=48; i <= 67; ++i) colors[i] = CV_RGB(255,0,0);    // 48 - 67 mouth           20 points
    for (int i=0; i < boxes.size(); ++i){
        for (int j = 0; j < landmarks[i].size(); ++j) {
            circle(image, landmarks[i][j], 2, CV_RGB(255, 0, 0), -1);
            //circle(image, landmarks[i][j], 2, colors[j], -1);
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

static void drawDistance(Mat& image, float distance) {
    stringstream outtext;
    outtext << "D: " << std::setprecision(3) << distance;
    cv::putText(image, outtext.str(), cv::Point(20, 100), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(255, 0, 0));
}


ResnetFaceDescriptor faceDescriptor;
thread recThread;
mutex recMutex;
Mat rgbaImage;
Rect faceBox;
vector<Point2f> faceLandmarks;
float faceDistance = 0;
bool recExit = false;

void recognize() {
    while (!recExit) {
        Mat img;
        Rect box;
        vector<Point2f> landmarks;

        recMutex.lock();
        img = rgbaImage;
        rgbaImage = Mat();
        box = faceBox;
        landmarks = faceLandmarks;
        recMutex.unlock();

        if (!img.data) {
            usleep(3000);
            continue;
        }

        static dlib::matrix<float,0,1> lastDescriptor;

        LOGD("face recongnition start");
        dlib::matrix<float,0,1> descriptor = faceDescriptor.extract(img, box, landmarks);
        LOGD("face recongnition end");

        if (lastDescriptor.size() < 1) {
            lastDescriptor = descriptor;
            continue;
        }
        faceDistance = faceDescriptor.distance(lastDescriptor, descriptor);
        LOGD("face distacne is %f", faceDistance);
    }
}

ImageProcessor::~ImageProcessor() {
    recExit = true;
    recThread.join();
}

bool ImageProcessor::init(const string& modelDir) {
    if (!mFaceDetector.load(modelDir)) {
        return false;
    }
    if (!mFaceLandmark.load(modelDir)) {
        return false;
    }
    if (!faceDescriptor.load(modelDir))
        return false;
    recExit = false;
    recThread = thread(recognize);
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

        // Do face recongnition
        if (recMutex.try_lock()) {
            image.copyTo(rgbaImage);
            faceBox = boxes[0];
            faceLandmarks = landmarks[0];
            recMutex.unlock();
        }
        drawDistance(image, faceDistance);
    }

}

