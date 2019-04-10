#include <opencv2/core.hpp>
#include "utils.h"
#include "imageprocessor.h"
#include "imageprocessor_jni.h"

using namespace cv;

//
// com.hsae.dms.ImageProcessor
//
JNIEXPORT jlong JNICALL Java_com_hsae_dms_ImageProcessor_nativeCreate(JNIEnv* env, jclass cls, jstring modelDir) {
    const char* dir = env->GetStringUTFChars(modelDir, nullptr);
    ImageProcessor* worker = new ImageProcessor();
    if (!worker->init(dir)) {
        delete worker;
        worker = nullptr;
    }
    env->ReleaseStringUTFChars(modelDir, dir);
    return reinterpret_cast<jlong>(worker);
}
JNIEXPORT void JNICALL Java_com_hsae_dms_ImageProcessor_nativeDestroy(JNIEnv* env, jclass cls, jlong handle) {
    delete reinterpret_cast<ImageProcessor*>(handle);
}
JNIEXPORT void JNICALL Java_com_hsae_dms_ImageProcessor_nativeProcess(JNIEnv *env, jclass cls,
    jlong handle, jobject byteBuffer, jint width, jint height, jint stride) {
    // The external data is not automatically de-allocated
    Mat image(height, width, CV_8UC4, env->GetDirectBufferAddress(byteBuffer), stride);
    // Process all face-related stuff
    ImageProcessor* worker = reinterpret_cast<ImageProcessor*>(handle);
    worker->process(image);
}

