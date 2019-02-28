#include <pthread.h>
#include "utils.h"

// NOTES:
// Each Android application owns only one JavaVM
// Save a global JavaVM for latter getting JavaEnv from different thread context.
static JavaVM* gJavaVM;
static pthread_key_t gThreadKey;

// NOTES:
// At thread exit, if a key value has a non-NULL destructor pointer, and the thread has a non-NULL value
// associated with that key, the value of the key is set to NULL, and then the function pointed to is called
// with the previously associated value as its sole argument
// see http://pubs.opengroup.org/onlinepubs/007904975/functions/pthread_key_create.html
static void onThreadExt(void* value) {
    gJavaVM->DetachCurrentThread();
}

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved) {
    LOGI("JNI_OnLoad");
    gJavaVM = vm;
    JNIEnv* env = nullptr;
    if (gJavaVM->GetEnv((void **)&env, JNI_VERSION_1_6) != JNI_OK) {
        LOGE("Failed to get the environment using GetEnv()");
        return -1;
    }
    if (pthread_key_create(&gThreadKey, onThreadExt)) {
        LOGE("Error initializing pthread key");
    }
    return JNI_VERSION_1_6;
}

// Any thread can safely call this method to get corresponding JavaEnv
JNIEnv* getJNIEnv() {
    JNIEnv* env = nullptr;
    int err = gJavaVM->AttachCurrentThread(&env, 0);
    if(err < 0) {
        LOGE("Failed to attach current thread");
        return 0;
    }
    if (!pthread_getspecific(gThreadKey)) {
        // Set the key, so that gJavaVM->DetachCurrentThread() would be called at thread exit.
        pthread_setspecific(gThreadKey, env);
    }
    return env;
}

cv::Rect toRect(jobject javaRect, JNIEnv* env) {
    if (!env) {
        env = getJNIEnv();
    }

    jclass rectClass = env->FindClass("android/graphics/Rect");
    jfieldID leftID = env->GetFieldID(rectClass, "left", "I");
    jfieldID topID = env->GetFieldID(rectClass, "top", "I");
    jfieldID rightID = env->GetFieldID(rectClass, "right", "I");
    jfieldID bottomID = env->GetFieldID(rectClass, "bottom", "I");

    cv::Rect rect;
    rect.x = env->GetIntField(javaRect, leftID);
    rect.y = env->GetIntField(javaRect, topID);
    rect.width = env->GetIntField(javaRect, rightID) - rect.x;
    rect.height = env->GetIntField(javaRect, bottomID) - rect.y;
    return rect;
}

jobjectArray newRectArray(const std::vector<cv::Rect>& rects, JNIEnv* env) {
    if (!env) {
        env = getJNIEnv();
    }

    // Rect class: android.graphics.Rect
    // Rect constructor: Rect(int left, int top, int right, int bottom);
    jclass rectClass = env->FindClass("android/graphics/Rect");
    jmethodID rectConstructor =  env->GetMethodID(rectClass, "<init>", "(IIII)V");
    jobject rectObject = env->NewObject(rectClass, rectConstructor, 0, 0, 0, 0);
    jobjectArray rectArray = env->NewObjectArray(rects.size(), rectClass, rectObject);
    for (int i=0; i < rects.size(); ++i) {
        jobject faceObject = env->NewObject(rectClass, rectConstructor,
                rects[i].x, rects[i].y, rects[i].x + rects[i].width, rects[i].y + rects[i].height);
        env->SetObjectArrayElement(rectArray, i, faceObject);
    }
    return rectArray;
}

jobjectArray newPointFArray(const std::vector<cv::Point2f>& points, JNIEnv* env) {
    if (!env) {
        env = getJNIEnv();
    }
    // RointF class: android.graphics.RointF
    // Rect constructor: RointF(float x, float y);
    jclass pointfClass = env->FindClass("android/graphics/PointF");
    jmethodID pointfConstructor =  env->GetMethodID(pointfClass, "<init>", "(FF)V");
    jobject pointfObject = env->NewObject(pointfClass, pointfConstructor, 0.0f, 0.0f);
    jobjectArray pointfArray = env->NewObjectArray(points.size(), pointfClass, pointfObject);
    for (int i=0; i < points.size(); ++i) {
        jobject mark = env->NewObject(pointfClass, pointfConstructor, points[i].x, points[i].y);
        env->SetObjectArrayElement(pointfArray, i, mark);
    }
    return pointfArray;
}

