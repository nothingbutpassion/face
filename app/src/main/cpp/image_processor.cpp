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
    const int d = 32;
    if (d > (right - left)/2 || d > (bottom - top)/2 )
        rectangle(frame, Point(left, top), Point(right, bottom), color);
    else {
        line(frame, Point(left, top), Point(left+d, top), color, 2, LINE_AA);
        line(frame, Point(left, top), Point(left, top+d), color, 2, LINE_AA);
        line(frame, Point(right, top), Point(right-d, top), color, 2, LINE_AA);
        line(frame, Point(right, top), Point(right, top+d), color, 2, LINE_AA);
        line(frame, Point(right, bottom), Point(right-d, bottom), color, 2, LINE_AA);
        line(frame, Point(right, bottom), Point(right, bottom-d), color, 2, LINE_AA);
        line(frame, Point(left, bottom), Point(left+d, bottom), color, 2, LINE_AA);
        line(frame, Point(left, bottom), Point(left, bottom-d), color, 2, LINE_AA);
    }
    if (label.size() > 0) {
        int baseLine;
        Size textSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 1, 2, &baseLine);
        int x0 = (frame.cols - textSize.width)/2;
        int y0 = baseLine + textSize.height;
        putText(frame, label, Point(x0, y0), FONT_HERSHEY_SIMPLEX, 1, color, 2);
    }
}

static void drawLandmarks(Mat& image, const vector<Point2f>& landmarks, const Scalar& color) {
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
        circle(image, landmarks[i], 2, color, -1);
        //circle(image, landmarks[i], 2, colors[i], -1);
    }
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

namespace {
    Rect adjustFaceBox(const Size& imgSize, const vector<Point2f>& landmarks) {
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

    Rect faceRoi(const Rect& lastFaceBox, const Size& imgSize) {
        Rect r = lastFaceBox;
        //LOGD("getRoi last_box=(%d, %d, %d, %d)", r.x, r.y, r.width, r.height);
        int dx = r.width*0.25;
        int dy = r.height*0.25;
        r.x -= dx;
        r.y -= dy;
        r.width += 2*dx;
        r.height += 2*dy;
        r.x = max(0, r.x);
        r.y = max(0, r.y);
        r.width = min(r.width, imgSize.width - r.x);
        r.height = min(r.height, imgSize.height - r.y);
        //LOGD("getRoi adjust_box=(%d, %d, %d, %d)", r.x, r.y, r.width, r.height);
        return r;
    }

    double faceScale(const Rect& lastFaceBox) {
        Rect r = lastFaceBox;
        //LOGD("getScale last_box=(%d, %d, %d, %d)", r.x, r.y, r.width, r.height);
        return 96.0/r.width;
        //LOGD("getScale scale=%.3f",  96.0/std::max(r.width, r.height));
    }

    float ear(const vector<Point2f>& pts) {
        Point2f py1 = pts[1] - pts[5];
        Point2f py2 = pts[2] - pts[4];
        Point2f px = pts[0] - pts[3];
        float d1 = sqrt(py1.x*py1.x + py1.y*py1.y);
        float d2 = sqrt(py2.x*py2.x + py2.y*py2.y);
        float d3 = sqrt(px.x*px.x + px.y*px.y);
        return 0.5*(d1 + d2)/d3;
    }

    float getEAR(const vector<Point2f>& landmarks) {
        // Left eye points: 36~41
        vector<Point2f> leftEyePts = {landmarks[36], landmarks[37], landmarks[38], landmarks[39], landmarks[40], landmarks[41]};
        // Right eye pooints: 42~47
        vector<Point2f> rightEyePts = {landmarks[42], landmarks[43], landmarks[44], landmarks[45], landmarks[46], landmarks[47]};
        float leftEAR = ear(leftEyePts);
        float rightEAR = ear(rightEyePts);
        return (leftEAR + rightEAR)/2;
    }

    float getMAR(const vector<Point2f>& landmarks) {
        Point2f py = (landmarks[50] + landmarks[51] + landmarks[52])/3 - (landmarks[56] + landmarks[57] + landmarks[58])/3;
        Point2f px = (landmarks[48] + landmarks[60])/2 - (landmarks[64] + landmarks[54])/2;
        float dy = sqrt(py.x*py.x + py.y*py.y);
        float dx = sqrt(px.x*px.x + px.y*px.y);
        return dy/dx;
    }

    uint64_t now() {
        timeval tv;
        gettimeofday(&tv, NULL);
        return tv.tv_sec*1000 + tv.tv_usec/1000;
    }

    struct Record {
        // timestamp
        uint64_t timestamp = 0;

        // face detection
        Rect box;
        float score = 0;
        int detector = 0;

        // EAR/MAR is computed based on landmarks
        float ear = 0;
        float smoothEar;
        float mar = 0;

        // head pose is computed based on landmarks
        Point3f angle;
        Point3f smoothAngle;

        vector<float> smoking;
        vector<float> calling;

    };

    struct DmsResult {
        bool smoking = false;
        bool calling = false;
        bool distracting = false;
        bool yawning = false;
        bool eyeclosed = false;
    };

    struct Recorder {
        static Recorder& instance() {
            static Recorder recorder;
            return recorder;
        }

        Record& last() {
            return records[records.size()-1];
        }

        void append(const Record& r) {
            constexpr int MAX_RECORDS = 128;
            constexpr int MAX_TIMEOUT = 5000;
            if (records.size() > MAX_RECORDS)
                records.erase(records.begin());
            while (records.size() > 0 && records[0].timestamp + MAX_TIMEOUT < r.timestamp)
                records.erase(records.begin());
            records.push_back(r);

            // Update smooth EAR/Angle
            constexpr int N = 5;
            int num = records.size() < N ? records.size() : N;
            float lastEar = (records.size() == 1) ? r.ear : records[records.size()-2].smoothEar;
            last().smoothEar = (r.ear + lastEar*(num-1))/num;
            LOGD("smooth ear: %.2f", last().smoothEar);

            Point3f lastAngle = (records.size() == 1) ? r.angle : records[records.size()-2].smoothAngle;
            last().smoothAngle = (r.angle + lastAngle*(num-1))/num;
            LOGD("smooth angle: %.2f, %.2f, %.2f", last().smoothAngle.x, last().smoothAngle.y, last().smoothAngle.z);
        }

        bool isSmoking() {
            constexpr int N = 4;
            if (records.size() < N)
                return false;
            bool smokingFeature1 = true;
            bool smokingFeature2 = true;
            // NOTES:
            // i is signed integer
            // records.size() is unsigned integer
            // Comparing signed with unsigned is prone to error
            for (int i=records.size() - N; i < records.size(); ++i) {
                if (records[i].smoking[0] < 0.6) {
                    smokingFeature1 = false;
                    break;
                }
            }
            for (int i=records.size() - N; i < records.size(); ++i) {
                if (records[i].smoking[1] < 0.95) {
                    smokingFeature2 = false;
                    break;
                }
            }
            // LOGI("smoking features: %d, %d", smokingFeature1, smokingFeature2);
            return (smokingFeature1 || smokingFeature2);
        }

        bool isCalling() {
            constexpr int N = 4;
            if (records.size() < N)
                return false;
            bool leftCalling = true;
            bool rightCalling = true;
            for (int i=records.size() - N; i < records.size(); ++i) {
                if (records[i].calling[0] < 0.6) {
                    leftCalling = false;
                    break;
                }
            }
            for (int i=records.size() - N; i < records.size(); ++i) {
                if (records[i].calling[1] < 0.6) {
                    rightCalling = false;
                    break;
                }
            }
            return (leftCalling && !rightCalling) || (!leftCalling && rightCalling);
        }

        bool isDistracting() {
            constexpr int N = 7;
            if (records.size() < N)
                return false;
            // NOTES:
            // Tested by head rotation angle
            // up:    180 -> 165  down: -180 -> -165
            // left:  0   -> +15  right:   0 -> -15
            bool upDownOk = true;
            bool leftRightOk = true;
            bool detectorOk = true;
            int detector = last().detector;
            for (int i=records.size() - N; i < records.size(); ++i) {
                if (abs(records[i].smoothAngle.x) > 160)
                    upDownOk = false;
                if (abs(records[i].smoothAngle.y) < 25)
                    leftRightOk = false;
                if (detector == 0 || detector == 3 || detector == 4 || records[i].detector != detector)
                    detectorOk = false;
            }
            if (detectorOk)
                return true;
            return false;
        }

        bool isEyeClosed() {
            constexpr int N = 9;
            constexpr float EAR_THRESHOLD = 0.18;
            if (records.size() < N) {
                return false;
            }
            for (int i=records.size() - N; i < records.size(); ++i) {
                if (records[i].smoothEar > EAR_THRESHOLD)
                    return false;
            }
            return true;
        }

        bool isYawn() {
            constexpr int N = 11;
            constexpr float MAR_THRESHOLD = 0.6;
            if (records.size() < N)
                return false;
            bool marOk = true;
            for (int i=records.size() - N; i < records.size(); ++i) {
                if (records[i].mar < MAR_THRESHOLD) {
                    marOk = false;
                    break;
                }
            }
            return marOk;
        }
    private:
        vector<Record> records;
    };


} // anonymous namespace


ImageProcessor::ImageProcessor() {}
ImageProcessor::~ImageProcessor() {}

bool ImageProcessor::init(const string& modelDir) {
    LOGI("Load models from %s ...", modelDir.c_str());
    if (!mFaceDetector.load(modelDir))
        return false;
    if (!mFaceLandmark.load(modelDir))
        return false;
    if (!mSmokingClassifier.load(modelDir))
        return false;
    if (!mCallingClassifier.load(modelDir))
        return false;
    LOGI("All model is loaded from %s", modelDir.c_str());
    return true;
}


void ImageProcessor::process(Mat& image) {
    // Save for process result
    DmsResult result;

    // Convert to single channel image
    Mat y;
    cvtColor(image, y, COLOR_RGBA2GRAY);

    // Face tracking
    static bool lastDetected = false;
    static Rect lastBox;
    double scale = 0.5;
    Rect roi(0, 0, y.cols, y.rows);
    Mat gray;
    if (lastDetected) {
        roi = faceRoi(lastBox, y.size());
        scale = faceScale(lastBox);
        resize(y(roi), gray, Size(roi.width*scale, roi.height*scale));
        LOGD("guess roi: (%d, %d, %d, %d) resize to: %d x %d", roi.x, roi.y, roi.width, roi.height, gray.cols, gray.rows);
    } else {
        Size s = y.size();
        if (scale != 1.0)
            resize(y, gray, Size(s.width*scale, s.height*scale));
        else
            gray = y;
        LOGD("reset roi: (%d, %d, %d, %d) resize to: %d x %d", roi.x, roi.y, roi.width, roi.height, gray.cols, gray.rows);
    }

    // Face detection
    vector<Rect> boxes;
    vector<float> scores;
    vector<int> indices;
    mFaceDetector.detect(gray, boxes, scores, &indices);
    if (boxes.size() == 0) {
        LOGD("no face detected");
        lastDetected = false;
        return;
    }
    lastDetected = true;
    int maxBoxIndex = 0;
    for (int i=1; i < boxes.size(); ++i)
        if (boxes[i].width*boxes[i].height > boxes[maxBoxIndex].width*boxes[maxBoxIndex].height)
            maxBoxIndex = i;
    Rect box = boxes[maxBoxIndex];
    box.x = roi.x + box.x/scale;
    box.y = roi.y + box.y/scale;
    box.width = box.width/scale;
    box.height= box.height/scale;
    LOGD("face box: (%d, %d, %d, %d), score: %.2f, detector: %d",
         box.x, box.y, box.width, box.height, scores[maxBoxIndex], indices[maxBoxIndex]);
    LOGD("face box: (%d, %d, %d, %d), score: %.2f, detector: %d",
         box.x, box.y, box.width, box.height, scores[maxBoxIndex], indices[maxBoxIndex]);

    // Landmark fit
    vector<Point2f> landmarks;
    mFaceLandmark.fit(y, box, landmarks);
    lastBox = adjustFaceBox(y.size(), landmarks);
    float ear = getEAR(landmarks);
    float mar = getMAR(landmarks);
    LOGD("face landmarks: %d, ear: %.2f, mar: %.2f", landmarks.size(), ear, mar);

    // Pose estimation
    Point3f pose;
    mPoseEstimator.estimate(y.size(), landmarks, pose);
    LOGD("head pose: (%.2f, %.2f, %.2f)", pose.x, pose.y, pose.z);

    // Smoking detection
    vector<float> smoking_results;
    Mat smoking_image = mSmokingClassifier.getInput(y, landmarks);
    mSmokingClassifier.predict(smoking_image, smoking_results);
    LOGD("smoking result: (%.2f, %.2f, %.2f)", smoking_results[0], smoking_results[1], smoking_results[2]);

    // Calling detection
    vector<float> left_calling_results;
    vector<float> right_calling_results;
    Mat left_calling_input = mCallingClassifier.getInput(y, landmarks, true);
    Mat right_calling_input = mCallingClassifier.getInput(y, landmarks, false);
    mCallingClassifier.predict(left_calling_input, left_calling_results);
    mCallingClassifier.predict(right_calling_input, right_calling_results);
    LOGD("calling result: (%.2f, %.2f)", left_calling_results[0], right_calling_results[0]);

    // Save history
    Record record;
    record.timestamp = now();
    record.box = lastBox;
    record.score = scores[maxBoxIndex];
    record.detector = indices[maxBoxIndex];
    record.ear = ear;
    record.mar = mar;
    record.angle = pose;
    record.smoking = vector<float>{smoking_results[0], smoking_results[1]};
    record.calling = vector<float>{left_calling_results[0], right_calling_results[0]};
    Recorder& recorder = Recorder::instance();
    recorder.append(record);

    // Get finial result
    result.smoking = recorder.isSmoking();
    result.calling = recorder.isCalling();
    result.distracting = recorder.isDistracting();
    result.yawning = recorder.isYawn();
    static uint64_t hotTime = 0;
    if (result.smoking || result.calling || result.distracting || result.yawning)
        hotTime = now();
    if (now() - hotTime < 2000)
        result.eyeclosed = false;
    else
        result.eyeclosed = recorder.isEyeClosed();

    // Output status
    Scalar color = CV_RGB(0, 255, 0);
    string label;
    if (result.smoking)
        label += " Smoking";
    if (result.calling)
        label += " Calling";
    if (result.yawning)
        label += " Yawning";
    if (result.distracting)
        label += " Distracting";
    if (result.eyeclosed)
        label += " Eye closed ";
    if (label != "") {
        color = CV_RGB(0, 0, 255);
    }
    drawBoxes(image, box.x, box.y, box.x + box.width, box.y + box.height, label, color);
    drawLandmarks(image, landmarks, color);
    drawAxis(image, mPoseEstimator.objectPoints(), mPoseEstimator.cameraMatrix(),
             mPoseEstimator.distCoeffs(), mPoseEstimator.rvec(), mPoseEstimator.tvec());

}

