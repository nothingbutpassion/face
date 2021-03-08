#include <opencv2/imgproc.hpp>
#include <tensorflow/lite/delegates/gpu/delegate.h>
#include <tensorflow/lite/c/c_api_experimental.h>
#include "tflite_smoking_classifier.h"
#include "utils.h"

#define LOG_TAG "TfLiteSmokingClassifier"

using namespace std;
using namespace cv;


TfLiteSmokingClassifier::~TfLiteSmokingClassifier() {
    release();
}

void TfLiteSmokingClassifier::release() {
    if (mInterpreter) {
        TfLiteInterpreterDelete(mInterpreter);
        mInterpreter = nullptr;
    }
    if (mDelegate) {
        TfLiteGpuDelegateV2Delete(mDelegate);
        mDelegate = nullptr;
    }
    if (mOptions) {
        TfLiteInterpreterOptionsDelete(mOptions);
        mInterpreter = nullptr;
    }
    if (mModel) {
        TfLiteModelDelete(mModel);
        mModel = nullptr;
    }
}

bool TfLiteSmokingClassifier::load(const string& modelDir, int numThreads, bool useGPU, bool useNNAPI) {
    string modelFile = modelDir + "/smoking_classifier.tflite";
    mModel = TfLiteModelCreateFromFile(modelFile.c_str());
    if (mModel == nullptr) {
        LOGE("TfLiteModelCreateFromFile failed: file=%s", modelFile.c_str());
        release();
        return false;
    }
    mOptions = TfLiteInterpreterOptionsCreate();
    if (numThreads)
        TfLiteInterpreterOptionsSetNumThreads(mOptions, numThreads);
    if (useGPU) {
        TfLiteGpuDelegateOptionsV2 options = TfLiteGpuDelegateOptionsV2Default();
        mDelegate = TfLiteGpuDelegateV2Create(&options);
        TfLiteInterpreterOptionsAddDelegate(mOptions, mDelegate);
    } else if (useNNAPI) {
        TfLiteInterpreterOptionsSetUseNNAPI(mOptions, true);
    }
    mInterpreter = TfLiteInterpreterCreate(mModel, mOptions);
    if (mInterpreter == nullptr) {
        LOGE("TfLiteInterpreterCreate failed");
        release();
        return false;
    }
    if (kTfLiteOk != TfLiteInterpreterAllocateTensors(mInterpreter)) {
        LOGE("TfLiteInterpreterAllocateTensors failed");
        release();
        return false;
    }
    TfLiteTensor* input = TfLiteInterpreterGetInputTensor(mInterpreter, 0);
    const TfLiteTensor* output = TfLiteInterpreterGetOutputTensor(mInterpreter, 0);
    if (input->type != kTfLiteFloat32 || input->dims->size != 4 || input->bytes != 64*64*4 ||
        input->dims->data[0] != 1 || input->dims->data[1] != 64 ||
        input->dims->data[2] != 64 || input->dims->data[3] != 1 ||
        output->type != kTfLiteFloat32 || output->dims->size != 2 || output->bytes != 3*4 ||
        output->dims->data[0] != 1 || output->dims->data[1] != 3) {
        LOGE("input tensor type must be float32 and shape must be (1,64,64,1),"
             "output tensor type must be float32 and shape must be (1,3)");
        release();
        return false;
    }
    return true;
}


void TfLiteSmokingClassifier::predict(const Mat& input, vector<float>& output) {
    int64_t t = getTickCount();
    TfLiteTensor* inputTensor = TfLiteInterpreterGetInputTensor(mInterpreter, 0);
    memcpy(inputTensor->data.f, input.data, inputTensor->bytes);
    TfLiteInterpreterInvoke(mInterpreter);
    const TfLiteTensor* outputTensor = TfLiteInterpreterGetOutputTensor(mInterpreter, 0);
    output.clear();
    for (int i=0; i < outputTensor->bytes/sizeof(float); i++)
        output.push_back(outputTensor->data.f[i]);
    LOGD("predict: %dms", int(double(getTickCount()-t)*1000/getTickFrequency()));
}

Mat TfLiteSmokingClassifier::getInput(const cv::Mat& gray, const vector<Point2f>& landmarks) {
    TfLiteTensor* inputTensor = TfLiteInterpreterGetInputTensor(mInterpreter, 0);
    Size size(inputTensor->dims->data[2], inputTensor->dims->data[1]);
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
    float w = 0.8*(x_max - x_min);
    float h = w;
    M.at<double>(0,2) -= (x0 - 0.5*w);
    M.at<double>(1,2) -= y0;
    M.at<double>(0,0) *= size.width/w;
    M.at<double>(0,1) *= size.width/w;
    M.at<double>(0,2) *= size.width/w;
    M.at<double>(1,0) *= size.height/h;
    M.at<double>(1,1) *= size.height/h;
    M.at<double>(1,2) *= size.height/h;
    Mat dst;
    warpAffine(gray, dst, M, size, INTER_CUBIC);
    Scalar mean;
    Scalar stddev;
    meanStdDev(dst, mean, stddev);
    stddev[0] += 1e-7;
    Mat input;
    dst.convertTo(input, CV_32F, 1./stddev[0], -mean[0]/stddev[0]);
    return input;
}
