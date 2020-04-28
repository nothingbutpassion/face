#include <opencv2/imgproc.hpp>
#include <tensorflow/lite/kernels/register.h>
#include "utils.h"
#include "tflite_action_classifier.h"

#define LOG_TAG "TFLiteActionModel"

using namespace std;
using namespace cv;
using namespace tflite;

bool TFLiteActionClassifier::load(const string& modelDir) {
    string modelFile = modelDir + "/action_classifier.tflite";
    mFlatBufferModel = FlatBufferModel::BuildFromFile(modelFile.c_str());
    if (mFlatBufferModel == nullptr) {
        LOGE("failed to load model files: %s", modelFile.c_str());
        return false;
    }
    ops::builtin::BuiltinOpResolver opResolver;
    tflite::InterpreterBuilder(*mFlatBufferModel, opResolver)(&mInterpreter);
    if(!mInterpreter){
        LOGE("failed to construct interpreter\n");
        return false;
    }
    if (mInterpreter->AllocateTensors() != kTfLiteOk) {
        LOGE("failed to allocate tensors\n");
        return false;
    }

    // Check input/output tensors
    const vector<int>& inputIndexes = mInterpreter->inputs();
    const vector<int>& outputIndexes = mInterpreter->outputs();
    if (inputIndexes.size() != 1 || outputIndexes.size() != 1) {
        LOGE("input tensors: %lu, output tensors: %lu, both must be 1", inputIndexes.size(), outputIndexes.size());
        return false;
    }
    TfLiteTensor* inputTensor = mInterpreter->tensor(inputIndexes[0]);
    TfLiteTensor* outputTensor = mInterpreter->tensor(outputIndexes[0]);
    if (inputTensor->type != kTfLiteFloat32 || inputTensor->dims->size != 4 ||
        inputTensor->dims->data[0] != 1 || inputTensor->dims->data[1] != 64 ||
        inputTensor->dims->data[2] != 64 || inputTensor->dims->data[3] != 3 ||
        outputTensor->type != kTfLiteFloat32 ||outputTensor->dims->size != 2 ||
        outputTensor->dims->data[0] != 1 || outputTensor->dims->data[1] != 3) {
        LOGE("input tensor type must be float32 and shape must be (1,64,64,3),"
             "output tensor type must be float32 and shape must be (1,3)");
        return false;
    }

    // Set the number of threads available to the interpreter.
    // mInterpreter->SetNumThreads(4);
    return true;
}


void TFLiteActionClassifier::predict(const Mat& image, vector<float>& results) {
    int64_t t = getTickCount();
    Mat input;
    resize(image, input, Size(64, 64));
    input.convertTo(input, CV_32F, 1/255.0, -0.5);
    TfLiteTensor* inputTensor = mInterpreter->tensor(mInterpreter->inputs()[0]);
    memcpy(inputTensor->data.f, input.data, 64*64*3*4);
    LOGI("Prepare inputs: %dms", int(double(getTickCount()-t)*1000/getTickFrequency()));

    // Do forward pass inference
    t = getTickCount();
    mInterpreter->Invoke();
    LOGI("Invoke inference: %dms", int(double(getTickCount()-t)*1000/getTickFrequency()));

    // Get outputs
    TfLiteTensor* outputTensor = mInterpreter->tensor(mInterpreter->outputs()[0]);
    float* detection = outputTensor->data.f;
    for (int i=0; i < outputTensor->dims->data[1]; ++i)
        results.push_back(detection[i]);
}