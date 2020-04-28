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

static void drawBoxes(const Mat& frame, int left, int top, int right, int bottom, const string& label, const Scalar& color) {
    rectangle(frame, Point(left, top), Point(right, bottom), color);

    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - labelSize.height), Point(left + labelSize.width, top + baseLine), color, FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar());
}

static void drawLandmarks(Mat& image, const vector<Rect>& boxes, const vector<Point2f>& landmark) {
//    vector<Scalar> colors(68);
//    for (int i=0; i <= 16; ++i) colors[i] =  CV_RGB(255,0,0);  // 0 - 16 is profile      17 points
//    for (int i=17; i <= 21; ++i) colors[i] = CV_RGB(255,0,0);  // 17 - 21 left eyebrow    5 points
//    for (int i=22; i <= 26; ++i) colors[i] = CV_RGB(255,0,0);  // 22 - 26 right eyebrow   5 points
//    for (int i=27; i <= 30; ++i) colors[i] = CV_RGB(255,0,0);  // 27 - 30 nose bridge     4 points
//    for (int i=31; i <= 35; ++i) colors[i] = CV_RGB(255,0,0);  // 31 - 35 nose hole       5 points
//    for (int i=36; i <= 41; ++i) colors[i] = CV_RGB(0,0,255);    // 36 - 41 left eye        6 points
//    for (int i=42; i <= 47; ++i) colors[i] = CV_RGB(0,0,255);    // 42 - 47 right eye       6 points
//    for (int i=48; i <= 67; ++i) colors[i] = CV_RGB(255,0,0);    // 48 - 67 mouth           20 points
    for (int i = 0; i < landmark.size(); ++i) {
        circle(image, landmark[i], 2, CV_RGB(255, 0, 0), -1);
        //circle(image, landmarks[i][j], 2, colors[j], -1);
    }
}

static void drawPoses(Mat& image, const vector<Point3f>& poses) {
    Point3f eulerAngle = poses[0];
    stringstream outtext;
    outtext << "x: " << std::setprecision(3) << eulerAngle.x;
    cv::putText(image, outtext.str(), cv::Point(4, 96), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(255, 0, 0));
    outtext.str("");
    outtext << "y: " << std::setprecision(3) << eulerAngle.y;
    cv::putText(image, outtext.str(), cv::Point(4, 120), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 0));
    outtext.str("");
    outtext << "z: " << std::setprecision(3) << eulerAngle.z;
    cv::putText(image, outtext.str(), cv::Point(4, 144), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 255));
    outtext.str("");
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

static void drawThumbnails(Mat& image, int currentPersion, vector<Mat>& persons) {
    constexpr int w = 64;
    constexpr int h = 64;
    int nx = image.cols/w;
    int px = (image.cols%w)/2;
    int ny = image.rows/h;
    int py = (image.rows%h)/2;
    for (int k=0; k < persons.size(); ++k) {
        int y = k/nx;
        int x = k%nx;
        px = px > 4 ? 4 : px;
        py = py > 4 ? 4 : py;
        Rect rect(px+x*w, py+y*h, w, h);
        Mat person;
        cvtColor(persons[k], person, COLOR_RGB2RGBA);
        resize(person, image(rect), Size(rect.width, rect.height));
        if (k == currentPersion)
            rectangle(image, rect, CV_RGB(0, 255, 0), 2);
    }
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
    void setFace(const Mat& img, const Rect& box, const vector<Point2f>& landmarks) {
        // LOGD("set image: %dx%d, box: (%d,%d,%d,%d)", img.cols, img.rows, box.x, box.y, box.width, box.height);
        lock_guard<mutex> lock(mMutex);
        cvtColor(img, mImage, COLOR_RGBA2RGB);
        mBox = box;
        mLandmarks = landmarks;
    }
    void getPersons(int& currentPersion, vector<Mat>& persons) {
        lock_guard<mutex> lock(mMutex);
        currentPersion = mCurrentPerson;
        for (auto p: mPersons)
            persons.push_back(get<1>(p));
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
            //LOGD("get image: %dx%d, box: (%d,%d,%d,%d)", img.cols, img.rows, box.x, box.y, box.width, box.height);
            LOGD("face recongnition start");
            Mat chip;
            Mat descriptor = mDescriptor.extract(img, box, landmarks, &chip);
            LOGD("face recongnition end");

            if (mPersons.size() == 0) {
                lock_guard<mutex> lock(mMutex);
                mPersons.push_back(make_tuple(descriptor, chip));
                mCurrentPerson = 0;
                LOGD("person %d detected", mCurrentPerson);
                continue;
            }
            int found = -1;
            float min = 1;
            for (int i=0; i < mPersons.size(); ++i) {
                float d = mDescriptor.distance(get<0>(mPersons[i]), descriptor);
                LOGD("distance with person %d: %f", i, d);
                if (d < 0.4 && d < min) {
                    min = d;
                    found = i;
                    mCurrentPerson = found;
                    LOGD("person %d matched (distance=%f)", mCurrentPerson, d);
                }
            }
            if (found != -1) {
                mPersons[found] = make_tuple(descriptor, chip);
            } else {
                lock_guard<mutex> lock(mMutex);
                mPersons.push_back(make_tuple(descriptor, chip));
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
    vector<tuple<Mat, Mat>>  mPersons;
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

//static string saveDir;
//static int numImages = 0;

bool ImageProcessor::init(const string& modelDir) {
    if (!mFaceDetector.load(modelDir))
        return false;
    if (!mFaceLandmark.load(modelDir))
        return false;
    if (!mActionClassifier.load(modelDir))
        return false;
    if (!FaceRecongnizer::instance().init(modelDir))
        return false;

    LOGD("model dir is %s", modelDir.c_str());
//    saveDir = modelDir;
    return true;
}

Rect action_box(const Size& imgSize, const Rect& faceBox) {
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
        vector<Point2f> landmarks;
        mFaceLandmark.fit(gray, boxes[0], landmarks);

        Mat rgba, bgr;
        Rect r = action_box(image.size(), boxes[0]);
        resize(image(r), rgba, Size(64, 64));
        cvtColor(rgba, bgr, COLOR_RGBA2BGR);
        vector<float> actions;
        mActionClassifier.predict(bgr, actions);

//        if (numImages < 100  &&
//           (actions[0] > actions[1] && actions[0] > actions[2] || actions[1] > actions[0] && actions[1] > actions[2])) {
//            string srcfile = saveDir + "/" + format("s_%03d_%.3f_%.3f_%.3f.jpg", numImages, actions[0], actions[1], actions[2]);
//            string actfile = saveDir + "/" + format("a_%03d_%.3f_%.3f_%.3f.jpg", numImages, actions[0], actions[1], actions[2]);
//            Mat srcimg;
//            cvtColor(image, srcimg, COLOR_RGBA2BGR);
//            bool srcOK = imwrite(srcfile.c_str(), srcimg);
//            bool actOK = imwrite(actfile.c_str(), bgr);
//            LOGD("save_images: status: %d,%d  path: %s,%s", srcOK, actOK, srcfile.c_str(), actfile.c_str());
//            numImages++;
//        }
//        // Face recongnition
//        if (0 <= boxes[0].x && boxes[0].x < image.cols &&
//            boxes[0].x + boxes[0].width < image.cols &&
//            0 <= boxes[0].y && boxes[0].y < image.rows &&
//            boxes[0].y + boxes[0].height < image.rows &&
//            indices[0] == 0)
//            FaceRecongnizer::instance().setFace(image, boxes[0], landmarks);
//
//        // Pose estimation
//        vector<Point3f> poses;
//        Mat rvec;
//        Mat tvec;
//        mPoseEstimator.estimate(image.size(), landmarks, poses, &rvec, &tvec);
//        drawPoses(image, poses);
//        drawAxis(image, mPoseEstimator.objectPoints(), mPoseEstimator.cameraMatrix(), mPoseEstimator.distCoeffs(), rvec, tvec);

//        // Draw thumbnails
//        int currentPerson;
//        vector<Mat> persons;
//        FaceRecongnizer::instance().getPersons(currentPerson, persons);
//        drawThumbnails(image, currentPerson, persons);

        // Draw face boxes
        //drawBoxes(image, boxes[0].x, boxes[0].y, boxes[0].x + boxes[0].width, boxes[0].y + boxes[0].height, scores[0], indices[0]);
        Scalar smoking = cv::Scalar(255, 0, 0, 0);
        Scalar calling = cv::Scalar(255, 255, 0, 0);
        Scalar normal = cv::Scalar(0, 255, 0, 0);
        Scalar color = normal;
        if (actions[0] > actions[1] && actions[0] > actions[2])
            color = smoking;
        else if (actions[1] > actions[0] && actions[1] > actions[2])
            color = calling;
        string label = format("actions: %.3f, %.3f, %.3f", actions[0], actions[1], actions[2]);
        drawBoxes(image, r.x, r.y, r.x + r.width, r.y + r.height, label, color);



        // Draw landmarks
        drawLandmarks(image, boxes, landmarks);
    }

}

