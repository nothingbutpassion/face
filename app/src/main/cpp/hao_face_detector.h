#ifndef HAO_FACE_DETECTOR_H
#define HAO_FACE_DETECTOR_H

#include <string>
#include <vector>
#include <memory>
#include <opencv2/core.hpp>
#include <tensorflow/lite/model.h>

class HaoFaceDetector {
public:
    bool load(const std::string& modelDir);
    void detect(const cv::Mat& image, std::vector<cv::Rect>& objects, std::vector<float>& confidences);

private:
    std::unique_ptr<tflite::FlatBufferModel> mFlatBufferModel;
    std::unique_ptr<tflite::Interpreter> mInterpreter;
};

#endif //HAO_FACE_DETECTOR_H
