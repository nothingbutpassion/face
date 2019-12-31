#include <opencv2/imgproc.hpp>
#include "utils.h"
#include "lbf_face_landmark.h"


#define LOG_TAG "LBFFaceLandmark"

using namespace std;
using namespace cv;

LBFFaceLandmark::LBFFaceLandmark() {
    mFacemark = face::FacemarkLBF::create(face::FacemarkLBF::Params());
}

bool LBFFaceLandmark::load(const string& modelDir) {
    string modelFile = modelDir + "/lbf_face_landmark.yaml";
    mFacemark->loadModel(modelFile);
    if (mFacemark->empty()) {
        LOGE("failed to load model file: %s", modelFile.c_str());
        return false;
    }
    return true;
}

bool LBFFaceLandmark::fit(const Mat& image, const vector<Rect>& faces, vector<vector<Point2f>>& landmarks) {
    bool ret = mFacemark->fit(image, faces, landmarks);
    if (!mFacemark->fit(image, faces, landmarks)) {
        LOGE("fit failed");
        return false;
    }
    return ret;
}

