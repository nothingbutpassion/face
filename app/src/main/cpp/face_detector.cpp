#include <opencv2/imgproc.hpp>
#include "utils.h"
#include "face_detector.h"

#undef  LOG_TAG
#define LOG_TAG "FaceDetector"

using namespace std;
using namespace cv;

bool FaceDetector::load(const string& modelFile) {
    if (!mFaceClassifier.load(modelFile)) {
        LOGE("failed to load model file: %s", modelFile.c_str());
        return false;
    }

    // Just for debugging
    int pos = modelFile.rfind("/");
    string modelDir = modelFile.substr(0, pos);
    mEyeClassifier.load(modelDir + "/haarcascade_eye.xml");
    mFaceLandmark.load(modelDir + "/lbfmodel.yaml");
    return true;
}

void FaceDetector::detect(const Mat& image, vector<Rect>& objects) {
    if(mFaceClassifier.empty()) {
        LOGE("model is empty!");
        return;
    }

    /**
    This function allows you to retrieve the final stage decision certainty of classification.
    For this, one needs to set `outputRejectLevels` on true and provide the `rejectLevels` and `levelWeights` parameter.
    For each resulting detection, `levelWeights` will then contain the certainty of classification at the final stage.
    This value can then be used to separate strong from weaker classifications.

    A code sample on how to use it efficiently can be found below:
    @code
    Mat img;
    vector<double> weights;
    vector<int> levels;
    vector<Rect> detections;
    CascadeClassifier model("/path/to/your/model.xml");
    model.detectMultiScale(img, detections, levels, weights, 1.1, 3, 0, Size(), Size(), true);
    cerr << "Detection " << detections[0] << " with weight " << weights[0] << endl;
    @endcode
    */
    vector<int> rejectLevels;
    vector<double> levelWeights;
    int64 t = getTickCount();
    mFaceClassifier.detectMultiScale(image, objects, rejectLevels, levelWeights, 1.1, 3, 0, Size(48, 48), Size(), true);
    double d = 1000*double(getTickCount() - t)/getTickFrequency();
    LOGI("detect faces: detected=%d size=%dx%d duration=%.1fms", objects.size(), image.cols, image.rows, d);

    // Just for debugging
    vector<Rect> faces;
    for (int i=0; i < objects.size(); ++i) {
        LOGI("detect faces: face=[%d,%d,%d,%d] level=%d weight=%f",
             objects[i].x, objects[i].y, objects[i].width, objects[i].height, rejectLevels[i], levelWeights[i]);

        Mat roi = image(objects[i]);
        vector<Rect> eyes;
        t = getTickCount();
        mEyeClassifier.detectMultiScale(roi, eyes, 1.1, 3, 0, Size(20, 20));
        d = 1000*double(getTickCount() - t)/getTickFrequency();
        LOGI("detect eyes: detected=%d, size=%dx%d duration=%.1fms", eyes.size(),  roi.cols, roi.rows, d);

        if (eyes.size() > 0) {
            int false_eye_count = 0;
            for (const Rect& eye: eyes) {
                if (eye.width > roi.cols/2 || eye.height > roi.rows/2 || eye.y > roi.rows/2) {
                    false_eye_count++;
                    LOGI("check eye: eye=[%d,%d,%d,%d], false=%d/total=%d",
                         eye.x, eye.y, eye.width, eye.height, false_eye_count, eyes.size());
                }
            }
            if (eyes.size() > false_eye_count) {
                LOGI("detect faces: face=[%d,%d,%d,%d] accepted",
                     objects[i].x, objects[i].y, objects[i].width, objects[i].height, rejectLevels[i], levelWeights[i]);
                rectangle(image, objects[i], CV_RGB(0, 0, 255), 1);
                faces.push_back(objects[i]);
            } else {
                LOGW("detect faces: face=[%d,%d,%d,%d] rejected (false eye detected)",
                     objects[i].x, objects[i].y, objects[i].width, objects[i].height, rejectLevels[i], levelWeights[i]);
            }
        } else {
            LOGW("detect faces: face=[%d,%d,%d,%d] rejected (no eyes detected)",
                 objects[i].x, objects[i].y, objects[i].width, objects[i].height, rejectLevels[i], levelWeights[i]);
        }
    }
    objects = std::move(faces);
}

bool FaceDetector::fit(const Mat& image, const Rect& face, vector<Point2f>& landmarks) {
    vector<Rect> faces = { face };
    vector<vector<Point2f>> marks = { landmarks };
    if (!mFaceLandmark.fit(image, faces, marks)) {
        return  false;
    }
    landmarks = std::move(marks[0]);
    return true;
}