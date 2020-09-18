#include <opencv2/imgproc.hpp>
#include "utils.h"
#include "caffe_smoking_classifier.h"

#define LOG_TAG "CaffeSmokingClassifier"

using namespace std;
using namespace cv;

bool CaffeSmokingClassifier::load(const string& modelDir) {
    string prototxt = modelDir + "/smoking_classifier.prototxt";
    string caffeModel = modelDir + "/smoking_classifier.caffemodel";
    mClassifierNet = dnn::readNetFromCaffe(prototxt, caffeModel);
    if (mClassifierNet.empty()) {
        LOGE("failed to load model files: %s, %s", prototxt.c_str(), caffeModel.c_str());
        return false;
    }
    return true;
}

void CaffeSmokingClassifier::predict(const Mat& gray, vector<float>& results) {

    // NOTES:
    // Take care of the 5th, 6th argument of cv::blobFromImage
    // Mat blobFromImage(InputArray image, double scalefactor=1.0, const Size& size = Size(),
    //                   const Scalar& mean = Scalar(), bool swapRB=false, bool crop=false,
    //                  int ddepth=CV_32F);
    Scalar mean;
    Scalar stddev;
    meanStdDev(gray, mean, stddev);
    double scale = 1.0/(stddev[0] + 1e-7);
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

Mat CaffeSmokingClassifier::getInput(const Mat& image, const vector<Point2f>& landmarks) {
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