#include <cmath>
#include <opencv2/imgproc.hpp>
#include <tensorflow/lite/kernels/register.h>
#include "utils.h"
#include "hao_face_detector.h"

#undef  LOG_TAG
#define LOG_TAG "HaoFaceDetector"

using namespace std;
using namespace cv;
using namespace tflite;

bool HaoFaceDetector::load(const string& modelDir) {
    string modelFile = modelDir + "/hao_face_detector.tflite";
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
        inputTensor->dims->data[0] != 1 || inputTensor->dims->data[1] != 160 ||
        inputTensor->dims->data[2] != 160 || inputTensor->dims->data[3] != 3 ||
        outputTensor->type != kTfLiteFloat32 ||outputTensor->dims->size != 3 ||
        outputTensor->dims->data[0] != 1 || outputTensor->dims->data[1] != 405 ||
        outputTensor->dims->data[2] != 6) {
        LOGE("input tensor type must be float32 and shape must be (1,160,160,3),"
              "output tensor type must be float32 and shape must be (1,405,6)");
        return false;
    }

    // Set the number of threads available to the interpreter.
//    mInterpreter->SetNumThreads(4);
    return true;
}


static Rect2f getDefaultBox(int index) {
    int sizes[] = {10, 5, 3, 1};
    float scales[] = {0.3f, 0.5f, 0.7f, 0.9f};
    float aspects[] = {0.5f, 0.8f, 1.0f};
    int nums[] = {3*10*10, 3*(10*10+5*5), 3*(10*10+5*5+3*3), 3*(10*10+5*5+3*3+1*1)};
    int i = 0;
    while (index >= nums[i])
        i++;
    if (i > 0)
        index -= nums[i-1];
    int rows = sizes[i];
    int cols = rows;
    int y = index/(3*cols);
    int x = (index - y*3*cols)/3;
    int k = index - y*3*cols - x*3;
    return Rect2f((x+0.5)/cols, (y+0.5)/rows, scales[i]*sqrt(aspects[k]), scales[i]/sqrt(aspects[k]));
}

static float iou(const Rect& b1, const Rect& b2) {
    int w = min(b1.x+b1.width, b2.x+b2.width) - max(b1.x, b2.x);
    int h = min(b1.y+b1.height, b2.y+b2.height) - max(b1.y, b2.y);
    if (w < 1 || h < 1)
        return 0.0f;
    return float(w*h)/(b1.area()+b2.area()- w*h);
}

static vector<int> nms(const vector<Rect>& boxes, const vector<float>& scores, float threshold) {
    vector<int> selected;
    vector<int> candidates;
    for (int i=0; i < scores.size(); ++i)
        candidates.push_back(i);

    while(candidates.size() > 0) {
        // Select the box with maximum score
        auto max = candidates.begin();
        for(auto it = candidates.begin() + 1; it != candidates.end(); ++it) {
            if (scores[*it] > scores[*max])
                max = it;
        }
        int maxIndex = *max;
        selected.push_back(maxIndex);
        candidates.erase(max);
        for(auto it = candidates.begin(); it != candidates.end(); ) {
            if (iou(boxes[maxIndex], boxes[*it]) > threshold)
                it = candidates.erase(it);
            else
                ++it;
        }
    }
    return selected;
}



void HaoFaceDetector::detect(const Mat& image, vector<Rect>& objects, vector<float>& confidences) {
    // Prepare for inputs
    Mat input;
    resize(image, input, Size(160,160));
    cvtColor(input, input, COLOR_RGBA2RGB);
    input.convertTo(input, CV_32F, 1/255.0, -0.5);
    TfLiteTensor* inputTensor = mInterpreter->tensor(mInterpreter->inputs()[0]);
    memcpy(inputTensor->data.f, input.data, 160*160*3*4);

    // Do forward pass inference
    mInterpreter->Invoke();

    // Get outputs
    TfLiteTensor* outputTensor = mInterpreter->tensor(mInterpreter->outputs()[0]);
    float* detection = outputTensor->data.f;
    int numBoxes = outputTensor->dims->data[1];
    int numProperties = outputTensor->dims->data[2];
    vector<Rect> faces;
    vector<float> scores;
    constexpr float confidenceThreshold = 0.5f;
    for (int i=0; i < numBoxes; ++i) {
        float c0 = detection[0];
        float c1 = detection[1];
        float p =  c0 > c1 ? 1/(1+exp(c1-c0)) : exp(c0-c1)/(1+exp(c0-c1));
        if (p > confidenceThreshold) {
            Rect2f d = getDefaultBox(i);
            float x = detection[2];
            float y = detection[3];
            float w = detection[4];
            float h = detection[5];
            Rect2f b( x*d.width+d.x, y*d.height+d.y, exp(w)*d.width, exp(h)*d.height);
            b.x -= b.width/2;
            b.y -= b.height/2;
            faces.push_back(Rect(
                int(b.x*image.cols + 0.5f), int(b.y*image.rows + 0.5f),
                int(b.width*image.cols + 0.5f), int(b.height*image.rows + 0.5f)));
            scores.push_back(p);
        }
        detection += numProperties;
    }

    constexpr float iouThreshold = 0.5f;
    vector<int> indexes = nms(faces, scores, iouThreshold);
    for (int i=0; i < indexes.size(); ++i) {
        objects.push_back(faces[indexes[i]]);
        confidences.push_back(scores[indexes[i]]);
    }
}

