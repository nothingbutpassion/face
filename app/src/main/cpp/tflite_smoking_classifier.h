#ifndef FACE_TFLITE_SMOKING_CLASSIFIER_H
#define FACE_TFLITE_SMOKING_CLASSIFIER_H

#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <tensorflow/lite/c/c_api.h>

class TfLiteSmokingClassifier {
public:
    ~TfLiteSmokingClassifier();
    bool load(const std::string& modelDir, int numThreads=0, bool useGPU = false, bool useNNAPI = false);
    void release();
    void predict(const cv::Mat& input, std::vector<float>& output);
    cv::Mat getInput(const cv::Mat& gray, const std::vector<cv::Point2f>& landmarks);
private:
    TfLiteModel* mModel = nullptr;
    TfLiteInterpreterOptions* mOptions = nullptr;
    TfLiteDelegate* mDelegate = nullptr;
    TfLiteInterpreter* mInterpreter = nullptr;
};

#endif //FACE_TFLITE_SMOKING_CLASSIFIER_H
