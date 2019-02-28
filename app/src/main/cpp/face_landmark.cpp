#include <opencv2/imgproc.hpp>
#include "utils.h"
#include "face_landmark.h"


#undef  LOG_TAG
#define LOG_TAG "FaceLandmark"

using namespace std;
using namespace cv;

FaceLandmark::FaceLandmark() {
    mFacemark = face::FacemarkLBF::create(face::FacemarkLBF::Params());
}

bool FaceLandmark::load(const string& modelDir) {
    string modelFile = modelDir + "/lbfmodel.yaml";
    mFacemark->loadModel(modelFile);
    if (mFacemark->empty()) {
        LOGE("failed to load model file: %s", modelFile.c_str());
        return false;
    }
    return true;
}

bool FaceLandmark::fit(const Mat& image, const vector<Rect>& faces, vector<vector<Point2f>>& landmarks) {
    bool ret = mFacemark->fit(image, faces, landmarks);
    if (!mFacemark->fit(image, faces, landmarks)) {
        LOGE("fit failed");
        return false;
    }
    // Just for debugging
    for (int i=0; i < faces.size(); ++i){
        for (int j = 0; j < landmarks[i].size(); ++j) {
            circle(image, landmarks[i][j], 2, CV_RGB(0, 255, 0), -1);
        }
    }
    return ret;
}

