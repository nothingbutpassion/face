#include "utils.h"
#include "caffe_action_classifier.h"

#define LOG_TAG "CaffeActionClassifier"

using namespace std;
using namespace cv;

bool CaffeActionClassifier::load(const string& modelDir) {
    string prototxt = modelDir + "/action_classifier.prototxt";
    string caffeModel = modelDir + "/action_classifier.caffemodel";
    mClassifierNet = dnn::readNetFromCaffe(prototxt, caffeModel);
    if (mClassifierNet.empty()) {
        LOGE("failed to load model files: %s, %s", prototxt.c_str(), caffeModel.c_str());
        return false;
    }
    return true;
}

void CaffeActionClassifier::predict(const Mat& bgr, vector<float>& results, const Scalar& meanBGR) {

    // NOTES:
    // Take care of the 5th, 6th argument of cv::blobFromImage
    // Mat blobFromImage(InputArray image, double scalefactor=1.0, const Size& size = Size(),
    //                   const Scalar& mean = Scalar(), bool swapRB=false, bool crop=false,
    //                  int ddepth=CV_32F);
    // v3a mean bgr: 91.22, 98.25, 117.24
    // v3b mean bgr: 90.06  93.66, 113.57
    // v3x mean bgr: 86.32  93.85, 113.96
    // mine bgr: 76, 89, 103
    Size inputSize(64, 64);
    Scalar trainBGR(91.22, 98.25, 117.24);
    Scalar mean = Scalar(127.5, 127.5, 127.5) + meanBGR - trainBGR;
    //Mat blob = dnn::blobFromImage(bgr, 1.0/255.0, inputSize, Scalar(127.5-15, 127.5-10, 127.5-14), false, false);
    Mat blob = dnn::blobFromImage(bgr, 1.0/255.0, inputSize, mean, false, false);
    mClassifierNet.setInput(blob);
    vector<Mat> outs;
    mClassifierNet.forward(outs);
    float* data = (float*)outs[0].data;
    for (int i = 0; i < outs[0].total(); ++i) {
        results.push_back(data[i]);
    }
}