#ifndef FACE_TFLITE_WRAPPER_H
#define FACE_TFLITE_WRAPPER_H

#include <tensorflow/lite/c/c_api.h>

#ifdef USE_ANDROID_LOG
    #include <android/log.h>
    #define LOGD(...)   ((void)__android_log_print(ANDROID_LOG_DEBUG, "tflite", __VA_ARGS__))
    #define LOGI(...)   ((void)__android_log_print(ANDROID_LOG_INFO,  "tflite", __VA_ARGS__))
    #define LOGW(...)   ((void)__android_log_print(ANDROID_LOG_WARN,  "tflite", __VA_ARGS__))
    #define LOGE(...)   ((void)__android_log_print(ANDROID_LOG_ERROR, "tflite", __VA_ARGS__))
#else
    #include <cstdio>
    #define LOGD(format, ...)   printf("[tflite] " format "\n", ## __VA_ARGS__)
    #define LOGI(format, ...)   printf("[tflite] " format "\n", ## __VA_ARGS__)
    #define LOGW(format, ...)   printf("[tflite] " format "\n", ## __VA_ARGS__)
    #define LOGE(format, ...)   printf("[tflite] " format "\n", ## __VA_ARGS__)
#endif

class TfLiteWrapper {
public:
    ~TfLiteWrapper();
    bool load(const char* modelFile, int numThreads, bool useNNAPI, bool useGPU, int gpuFlags);
    void release();
    void forward();
    void setRandomInput();
    int inputCount();
    int inputBytes(int index);
    int inputWidth(int index);
    int inputHeight(int index);
    void setInput(int index, const void* data, int size);
    int outputCount();
    int outputBytes(int index);
    void getOutput(int index, void* data, int size);


private:
    TfLiteModel* mModel = nullptr;
    TfLiteInterpreterOptions* mOptions = nullptr;
    TfLiteDelegate* mDelegate = nullptr;
    TfLiteInterpreter* mInterpreter = nullptr;
};

#endif //FACE_TFLITE_WRAPPER_H
