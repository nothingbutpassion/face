#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include "utils.h"
#include "kazemi_face_landmark.h"

#define LOG_TAG "KazemiFaceLandmark"

KazemiFaceLandmark::~KazemiFaceLandmark() {
    delete static_cast<dlib::shape_predictor*>(mShapePredictor);
}

bool KazemiFaceLandmark::load(const std::string& modelDir) {
    std::string modelFile = modelDir + "/kazemi_face_landmark.dat";
    try {
        dlib::shape_predictor* predictor = new dlib::shape_predictor();
        dlib::deserialize(modelFile) >> *predictor;
        mShapePredictor = predictor;
    } catch (std::exception& e) {
        LOGE("failed to load %s with exception: %s", modelFile.c_str(),  e.what());
        return false;
    }
    return true;
}

void KazemiFaceLandmark::fit(const cv::Mat& gray, const cv::Rect& face, std::vector<cv::Point2f>& landmarks) {
    dlib::cv_image<uchar> cimg(gray);
    std::vector<dlib::full_object_detection> shapes;
    dlib::rectangle rect(face.x, face.y, face.x + face.width, face.y + face.height);
    dlib::full_object_detection shape = (*static_cast<dlib::shape_predictor*>(mShapePredictor))(cimg, rect);
    for (int i=0; i < shape.num_parts(); ++i) {
        dlib::point pt = shape.part(i);
        landmarks.push_back(cv::Point2f(pt.x(), pt.y()));
    }
}

