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

void CaffeActionClassifier::predict(const Mat& bgr, vector<float>& results) {

    // NOTES:
    // Take care of the 5th, 6th argument of cv::blobFromImage
    // Mat blobFromImage(InputArray image, double scalefactor=1.0, const Size& size = Size(),
    //                   const Scalar& mean = Scalar(), bool swapRB=false, bool crop=false,
    //                  int ddepth=CV_32F);
    Mat blob = dnn::blobFromImage(bgr, 1.0/255.0, Size(64, 64), Scalar(127.5, 127.5, 127.5), false, false);
    mClassifierNet.setInput(blob);
    vector<Mat> outs;
    mClassifierNet.forward(outs);
    float* data = (float*)outs[0].data;
    for (int i = 0; i < outs[0].total(); ++i) {
        results.push_back(data[i]);
    }
}