#include <unistd.h>
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

static void drawBoxes(const Mat& frame, float confidence, int left, int top, int right, int bottom) {
    rectangle(frame, Point(left, top), Point(right, bottom), CV_RGB(0, 255, 0));
    string label = format("%.2f", confidence);

    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - labelSize.height), Point(left + labelSize.width, top + baseLine), CV_RGB(0, 255, 0), FILLED);
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


struct FaceRecongnizer {
    static FaceRecongnizer& instance() {
        static FaceRecongnizer recongnizer;
        return recongnizer;
    }
    bool init(const string& modelDir) {
        mInit = mDescriptor.load(modelDir);
        if (mInit)
            start();
        return mInit;
    }
    void start() {
        mThreadExit = false;
        mThread = thread(&FaceRecongnizer::recognize, this);
    }
    void stop() {
        mThreadExit = true;
        if (mInit)
            mThread.join();
    }
    void set(const Mat& img, const Rect& box, const vector<Point2f>& landmarks) {
        LOGD("set image: %dx%d, box: (%d,%d,%d,%d)", img.cols, img.rows, box.x, box.y, box.width, box.height);
        lock_guard<mutex> lock(mMutex);
        img.copyTo(mImage);
        mBox = box;
        mLandmarks = landmarks;
    }
    void recognize() {
        while (!mThreadExit) {
            Mat img;
            Rect box;
            vector<Point2f> landmarks;

            mMutex.lock();
            img = mImage;
            mImage = Mat();
            box = mBox;
            landmarks = mLandmarks;
            mMutex.unlock();

            if (!img.data) {
                usleep(100000);
                continue;
            }
            LOGD("get image: %dx%d, box: (%d,%d,%d,%d)", img.cols, img.rows, box.x, box.y, box.width, box.height);
            static dlib::matrix<float,0,1> lastDescriptor;
            LOGD("face recongnition start");
            dlib::matrix<float,0,1> descriptor = mDescriptor.extract(img, box, landmarks);
            LOGD("face recongnition end");

            if (mPersons.size() == 0) {
                Mat thumbnail;
                img(box).copyTo(thumbnail);
                mPersons.push_back(make_tuple(descriptor, thumbnail));
                mCurrentPerson = 0;
                LOGD("person %d detected", mCurrentPerson);
                continue;
            }
            int found = -1;
            for (int i=0; i < mPersons.size(); ++i) {
                float d = mDescriptor.distance(get<0>(mPersons[i]), descriptor);
                if (d < 0.6) {
                    found = i;
                    mCurrentPerson = found;
                    LOGD("person %d matched (distance=%f)", mCurrentPerson, d);
                    break;
                }
            }
            if (found == -1) {
                Mat thumbnail;
                img(box).copyTo(thumbnail);
                mPersons.push_back(make_tuple(descriptor, thumbnail));
                mCurrentPerson = mPersons.size() - 1;
                LOGD("new person %d detected", mCurrentPerson);
            }
        }
    }
private:
    ResnetFaceDescriptor mDescriptor;
    bool mInit = false;
    Mat mImage;
    Rect mBox;
    vector<Point2f> mLandmarks;
    vector<tuple<dlib::matrix<float,0,1>, Mat>>  mPersons;
    int mCurrentPerson = -1;
    mutex mMutex;
    thread mThread;
    bool mThreadExit = false;
};


ImageProcessor::ImageProcessor() {
}

ImageProcessor::~ImageProcessor() {
    FaceRecongnizer::instance().stop();
}

bool ImageProcessor::init(const string& modelDir) {
    if (!mFaceDetector.load(modelDir))
        return false;
    if (!mFaceLandmark.load(modelDir))
        return false;
    if (!FaceRecongnizer::instance().init(modelDir))
        return false;
    return true;
}

void ImageProcessor::process(Mat& image) {
    vector<Rect> boxes;
    vector<float> confidences;
    // Face detection
    mFaceDetector.detect(image, boxes, confidences);
    if (boxes.size() > 0) {
        for (int i=0; i < boxes.size(); ++i)
            drawBoxes(image, confidences[i], boxes[i].x, boxes[i].y, boxes[i].x + boxes[i].width, boxes[i].y + boxes[i].height);

        // Face landmarks
        vector<vector<Point2f>> landmarks;
        mFaceLandmark.fit(image, boxes, landmarks);
        drawLandmarks(image, boxes, landmarks);

        // Pose estimation
        vector<Point3f> poses;
        mPoseEstimator.estimate(image, landmarks[0], poses);
        drawPoses(image, poses);

        // Face recongnition
        if (0 <= boxes[0].x && boxes[0].x < image.cols &&
            boxes[0].x + boxes[0].width < image.cols &&
            0 <= boxes[0].y && boxes[0].y < image.rows &&
            boxes[0].y + boxes[0].height < image.rows)
                FaceRecongnizer::instance().set(image, boxes[0], landmarks[0]);
        //drawDistance(image, FaceRecongnizer::instance().distance());
    }
}

