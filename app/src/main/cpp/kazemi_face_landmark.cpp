#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include "utils.h"
#include "kazemi_face_landmark.h"

#define LOG_TAG "KazemiFaceLandmark"

using namespace std;
using namespace cv;
using namespace dlib;

bool KazemiFaceLandmark::load(const string& modelDir) {
    string modelFile = modelDir + "/kazemi_face_landmark.dat";
    try {
        deserialize(modelFile) >> mShapePredictor;
    } catch (exception& e) {
        LOGE("exception: %s", e.what());
        LOGE("failed to load model file: %s", modelFile.c_str());
        return false;
    }
    return true;
}

bool KazemiFaceLandmark::fit(const cv::Mat& image, const std::vector<Rect>& faces, std::vector<std::vector<Point2f>>& landmarks)  {
    cv::Mat gray;
    cvtColor(image, gray, COLOR_RGBA2GRAY);
    cv_image<uchar> cimg(gray);
    for (auto& face: faces) {
        std::vector<full_object_detection> shapes;
        dlib::rectangle rect(face.x, face.y, face.x + face.width, face.y + face.height);
        full_object_detection shape = mShapePredictor(cimg, rect);
        int numPoints = shape.num_parts();
        if (numPoints > 0) {
            std::vector<Point2f> points;
            for (int i=0; i < numPoints; ++i) {
                dlib::point pt = shape.part(i);
                points.push_back(Point2f(pt.x(), pt.y()));
            }
            landmarks.push_back(points);
        }
    }
    return true;
}
