#include <opencv2/imgproc.hpp>
#include "utils.h"
#include "caffe_calling_classifier.h"

#define LOG_TAG "CaffeCallingClassifier"

using namespace std;
using namespace cv;

bool CaffeCallingClassifier::load(const string& modelDir) {
    string prototxt = modelDir + "/calling_classifier.prototxt";
    string caffeModel = modelDir + "/calling_classifier.caffemodel";
    mClassifierNet = dnn::readNetFromCaffe(prototxt, caffeModel);
    if (mClassifierNet.empty()) {
        printf("failed to load model files: %s, %s\n", prototxt.c_str(), caffeModel.c_str());
        return false;
    }
    return true;
}
void CaffeCallingClassifier::predict(const Mat& gray, vector<float>& results) {
    Scalar mean;
    Scalar stddev;
    meanStdDev(gray, mean, stddev);
    double scale = 1.0 / stddev[0];
    Size inputSize(64, 64);
    Mat blob = dnn::blobFromImage(gray, scale, inputSize, mean, false, false);
    mClassifierNet.setInput(blob);
    vector<Mat> outs;
    mClassifierNet.forward(outs);
    float* data = (float*)outs[0].data;
    for (int i = 0; i < outs[0].total(); ++i) {
        results.push_back(data[i]);
    }
}

Mat CaffeCallingClassifier::getInput(const Mat& image, const vector<Point2f>& landmarks) {
    Point2f leftEye(0, 0);
    Point2f rightEye(0, 0);
    for (int i = 42; i < 48; ++i)
        leftEye += landmarks[i];
    for (int i = 36; i < 42; ++i)
        rightEye += landmarks[i];
    Point2f center((rightEye.x + leftEye.x) / 12, (rightEye.y + leftEye.y) / 12);
    float angle = atan2f(rightEye.y - leftEye.y, rightEye.x - leftEye.x) * 180 / 3.1415926535 + 180;
    Mat M = getRotationMatrix2D(center, angle, 1.0);
    vector<Point2f> marks;
    float x_max = -1000000;
    float x_min = 1000000;
    for (const Point2f& p : landmarks) {
        float x = p.x*M.at<double>(0, 0) + p.y*M.at<double>(0, 1) + M.at<double>(0, 2);
        float y = p.x*M.at<double>(1, 0) + p.y*M.at<double>(1, 1) + M.at<double>(1, 2);
        x_max = max(x, x_max);
        x_min = min(x, x_min);
        marks.push_back(Point2f(x, y));
    }
    float x0 = (marks[27].x + marks[28].x + marks[29].x + marks[30].x + marks[8].x) / 5;
    float y0 = (marks[36].y + marks[39].y + marks[42].y + marks[45].y) / 4;
    float h = marks[8].y - y0 + (marks[8].y - marks[19].y) / 3;
    float w = h;
    M.at<double>(0, 2) -= (x0 - w);
    M.at<double>(1, 2) -= y0;
    M.at<double>(0, 0) *= 64 / w;
    M.at<double>(0, 1) *= 64 / w;
    M.at<double>(0, 2) *= 64 / w;
    M.at<double>(1, 0) *= 64 / h;
    M.at<double>(1, 1) *= 64 / h;
    M.at<double>(1, 2) *= 64 / h;
    Mat dst;
    int64 t = getTickCount();
    warpAffine(image, dst, M, Size(128, 64));
    return dst;
}