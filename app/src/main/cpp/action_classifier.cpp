#include "utils.h"
#include "action_classifier.h"

#define LOG_TAG "ActionClassifier"

using namespace std;
using namespace cv;

//static vector<String> getOutputsNames(const dnn::Net& net) {
//    static std::vector<String> names;
//    if (names.empty()) {
//        vector<int> outLayers = net.getUnconnectedOutLayers();
//        vector<String> layersNames = net.getLayerNames();
//        for (int i=0; i < layersNames.size(); ++i)
//            LOGD("layer：%s", layersNames[i].c_str());
//        names.resize(outLayers.size());
//        for (size_t i = 0; i < outLayers.size(); ++i) {
//            names[i] = layersNames[outLayers[i] - 1];
//            LOGD("output layer index: %d, name: %s", outLayers[i] - 1, names[i].c_str());
//        }
//    }
//    return names;
//}

bool ActionClassifier::load(const string& modelDir) {
    string prototxt = modelDir + "/action_classifier.prototxt";
    string caffeModel = modelDir + "/action_classifier.caffemodel";
    mClassifierNet = dnn::readNetFromCaffe(prototxt, caffeModel);
    if (mClassifierNet.empty()) {
        LOGE("failed to load model files: %s, %s", prototxt.c_str(), caffeModel.c_str());
        return false;
    }
    return true;
}

void ActionClassifier::predict(const Mat& bgr, vector<float>& prediction) {
    vector<Mat> outs;
    // NOTES:
    // Take care of the 5th, 6th argument of this function
    //   Mat blobFromImage(InputArray image, double scalefactor=1.0, const Size& size = Size(),
    //       const Scalar& mean = Scalar(), bool swapRB=true, bool crop=true, int ddepth=CV_32F);
    Mat blob = dnn::blobFromImage(bgr, 1.0/255.0, Size(64, 64), Scalar(127.5, 127.5, 127.5), false, false);
    mClassifierNet.setInput(blob);
    mClassifierNet.forward(outs);
    float* data = (float*)outs[0].data;
    for (int i = 0; i < outs[0].total(); ++i) {
        prediction.push_back(data[i]);
    }
}