#ifndef UTILS_H
#define UTILS_H

#include <jni.h>
#include <vector>
#include <opencv2/core.hpp>

#define DEBUG_FACE
#ifdef  DEBUG_FACE
    #include <android/log.h>
    #define LOG_TAG     "Face"
    #define LOGD(...)   ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))
    #define LOGI(...)   ((void)__android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__))
    #define LOGW(...)   ((void)__android_log_print(ANDROID_LOG_WARN,  LOG_TAG, __VA_ARGS__))
    #define LOGE(...)   ((void)__android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__))
#else
    #define LOGD(...)
    #define LOGI(...)
    #define LOGW(...)
    #define LOGE(...)
#endif


JNIEnv* getJNIEnv();
cv::Rect toRect(jobject javaRect, JNIEnv* env = nullptr);
jobjectArray newRectArray(const std::vector<cv::Rect>& rects, JNIEnv* env = nullptr);
jobjectArray newPointFArray(const std::vector<cv::Point2f>& points, JNIEnv* env = nullptr);



#endif // UTILS_H
