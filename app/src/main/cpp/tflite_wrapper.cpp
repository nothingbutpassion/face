#include <string>
#include <tensorflow/lite/delegates/gpu/delegate.h>
#include <tensorflow/lite/c/c_api_experimental.h>
#include "tflite_wrapper.h"


TfLiteWrapper::~TfLiteWrapper() {
    release();
}

//typedef enum {
//    kTfLiteNoType = 0,
//    kTfLiteFloat32 = 1,
//    kTfLiteInt32 = 2,
//    kTfLiteUInt8 = 3,
//    kTfLiteInt64 = 4,
//    kTfLiteString = 5,
//    kTfLiteBool = 6,
//    kTfLiteInt16 = 7,
//    kTfLiteComplex64 = 8,
//    kTfLiteInt8 = 9,
//    kTfLiteFloat16 = 10,
//    kTfLiteFloat64 = 11,
//} TfLiteType;
static const char* types[] = {
    "kTfLiteNoType",
    "kTfLiteFloat32",
    "kTfLiteInt32",
    "kTfLiteUInt8",
    "kTfLiteInt64",
    "kTfLiteString",
    "kTfLiteBool",
    "kTfLiteInt16",
    "kTfLiteComplex64",
    "kTfLiteInt8",
    "kTfLiteFloat16",
    "kTfLiteFloat64",
};


static void dumpTensor(const TfLiteTensor* tensor, const char* prefix = "") {
    std::string str = "tensor(type=";
    str += types[tensor->type];
    str += ", shape=(";
    for (int i=0; i < tensor->dims->size; i++) {
        if (i != 0)
            str += ", ";
        str += std::to_string(tensor->dims->data[i]);
    }
    str += "))";
    LOGI("%s%s", prefix, str.c_str());
}

bool TfLiteWrapper::load(const char* modelFile, int numThreads, bool useNNAPI, bool useGPU, int gpuFlags) {
    mModel = TfLiteModelCreateFromFile(modelFile);
    if (mModel == nullptr) {
        LOGE("TfLiteModelCreateFromFile failed: file=%s", modelFile);
        release();
        return false;
    }
    mOptions = TfLiteInterpreterOptionsCreate();
    if (numThreads)
        TfLiteInterpreterOptionsSetNumThreads(mOptions, numThreads);
    if (useGPU) {
        TfLiteGpuDelegateOptionsV2 options = TfLiteGpuDelegateOptionsV2Default();
        LOGI("original gpu experimental flags: 0x%x", options.experimental_flags);
        if (gpuFlags)
            options.experimental_flags = gpuFlags;
        LOGI("current gpu experimental flags: 0x%x", options.experimental_flags);
        mDelegate = TfLiteGpuDelegateV2Create(&options);
        TfLiteInterpreterOptionsAddDelegate(mOptions, mDelegate);
    } else if (useNNAPI) {
        TfLiteInterpreterOptionsSetUseNNAPI(mOptions, true);
    }
    mInterpreter = TfLiteInterpreterCreate(mModel, mOptions);
    if (mInterpreter == nullptr) {
        LOGE("TfLiteInterpreterCreate failed");
        release();
        return false;
    }
    if (kTfLiteOk != TfLiteInterpreterAllocateTensors(mInterpreter)) {
        LOGE("TfLiteInterpreterAllocateTensors failed");
        release();
        return false;
    }

    bool supported = true;
    int input_count = TfLiteInterpreterGetInputTensorCount(mInterpreter);
    for (int i=0; i < input_count; ++i) {
        TfLiteTensor* tensor = TfLiteInterpreterGetInputTensor(mInterpreter, i);
        dumpTensor(tensor, "input:  ");
        if (tensor->type != kTfLiteFloat32 || tensor->dims->size != 4)
            supported = false;
    }
    int output_count = TfLiteInterpreterGetInputTensorCount(mInterpreter);
    for (int i=0; i < output_count; ++i) {
        const TfLiteTensor* tensor = TfLiteInterpreterGetOutputTensor(mInterpreter, i);
        dumpTensor(TfLiteInterpreterGetOutputTensor(mInterpreter, i), "output: ");
        if (tensor->type != kTfLiteFloat32)
            supported = false;
    }
    if (!supported) {
        LOGE("only float input/output tensors and each input tensor dims must be 4");
        release();
        return false;
    }
    return true;
}


void TfLiteWrapper::release() {
    if (mInterpreter) {
        TfLiteInterpreterDelete(mInterpreter);
        mInterpreter = nullptr;
    }
    if (mDelegate) {
        TfLiteGpuDelegateV2Delete(mDelegate);
        mDelegate = nullptr;
    }
    if (mOptions) {
        TfLiteInterpreterOptionsDelete(mOptions);
        mInterpreter = nullptr;
    }
    if (mModel) {
        TfLiteModelDelete(mModel);
        mModel = nullptr;
    }
}

void TfLiteWrapper::forward() {
    TfLiteInterpreterInvoke(mInterpreter);
}

void TfLiteWrapper::setRandomInput() {
    int input_count = TfLiteInterpreterGetInputTensorCount(mInterpreter);
    for (int i=0; i < input_count; ++i) {
        TfLiteTensor* tensor = TfLiteInterpreterGetInputTensor(mInterpreter, i);
        for (int k=0; k < tensor->bytes/sizeof(float); ++k)
            tensor->data.f[k] = 2.f*float(rand())/RAND_MAX - 1.f;
    }
}


int TfLiteWrapper::inputCount() {
    return TfLiteInterpreterGetInputTensorCount(mInterpreter);
}

int TfLiteWrapper::inputBytes(int index) {
    return TfLiteInterpreterGetInputTensor(mInterpreter, index)->bytes;
}

int TfLiteWrapper::inputWidth(int index) {
    return TfLiteInterpreterGetInputTensor(mInterpreter, index)->dims->data[2];
}
int TfLiteWrapper::inputHeight(int index) {
    return TfLiteInterpreterGetInputTensor(mInterpreter, index)->dims->data[1];
}

void TfLiteWrapper::setInput(int index, const void* data, int size) {
    TfLiteTensor* tensor = TfLiteInterpreterGetInputTensor(mInterpreter, index);
    TfLiteTensorCopyFromBuffer(tensor, data, size);
}

int TfLiteWrapper::outputCount() {
    return TfLiteInterpreterGetOutputTensorCount(mInterpreter);
}
int TfLiteWrapper::outputBytes(int index) {
    return TfLiteInterpreterGetOutputTensor(mInterpreter, index)->bytes;
}
void TfLiteWrapper::getOutput(int index, void* data, int size) {
    const TfLiteTensor* tensor = TfLiteInterpreterGetOutputTensor(mInterpreter, index);
    TfLiteTensorCopyToBuffer(tensor, data, size);
}



