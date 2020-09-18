#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include "hog_face_detector.h"

HOGFaceDetector::~HOGFaceDetector() {
    delete static_cast<dlib::frontal_face_detector*>(mDetector);
}

bool HOGFaceDetector::load(const std::string& modelDir) {
    //  NOTES: model data is embedded in code, no need to load
    if (!mDetector) {
        dlib::frontal_face_detector* detector = new dlib::frontal_face_detector();
        *detector = dlib::get_frontal_face_detector();
        mDetector = detector;
    }
    return (mDetector != nullptr);
}



void HOGFaceDetector::detect(const cv::Mat& gray, std::vector<cv::Rect>& faces, std::vector<float>& scores, std::vector<int>* indices) {
    dlib::cv_image<uchar> cimg(gray);
    std::vector<dlib::rect_detection> dets;
    (*static_cast<dlib::frontal_face_detector*>(mDetector))(cimg, dets);
    for (dlib::rect_detection& d: dets) {
        faces.push_back(cv::Rect(d.rect.left(), d.rect.top(), d.rect.width(), d.rect.height()));
        scores.push_back(d.detection_confidence);
        if (indices)
            indices->push_back(d.weight_index);
    }
}

