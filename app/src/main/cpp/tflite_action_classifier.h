#ifndef FACE_TFLITE_ACTION_CLASSIFIER_H
#define FACE_TFLITE_ACTION_CLASSIFIER_H

#include <string>
#include <vector>
#include <memory>
#include <opencv2/core.hpp>
#include <tensorflow/lite/model.h>

class TFLiteActionClassifier {
public:
    bool load(const std::string& modelDir);
    void predict(const cv::Mat& image, std::vector<float>& results);

private:
    std::unique_ptr<tflite::FlatBufferModel> mFlatBufferModel;
    std::unique_ptr<tflite::Interpreter> mInterpreter;
};


#endif //FACE_TFLITE_ACTION_CLASSIFIER_H
