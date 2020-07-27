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
    double scale = 1.0/stddev[0];
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
