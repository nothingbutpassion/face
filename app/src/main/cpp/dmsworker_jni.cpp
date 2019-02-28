#include <opencv2/core.hpp>
#include "utils.h"
#include "dmsworker.h"
#include "dmsworker_jni.h"

using namespace cv;

//
// com.hsae.dms.DMSWorker
//
JNIEXPORT jlong JNICALL Java_com_hsae_dms_DMSWorker_nativeCreate(JNIEnv* env, jclass cls, jstring modelDir) {
    const char* dir = env->GetStringUTFChars(modelDir, nullptr);
    DMSWorker* worker = new DMSWorker();
    if (!worker->init(dir)) {
        delete worker;
        worker = nullptr;
    }
    env->ReleaseStringUTFChars(modelDir, dir);
    return reinterpret_cast<jlong>(worker);
}
JNIEXPORT void JNICALL Java_com_hsae_dms_DMSWorker_nativeDestroy(JNIEnv* env, jclass cls, jlong handle) {
    delete reinterpret_cast<FaceDetector*>(handle);
}
JNIEXPORT void JNICALL Java_com_hsae_dms_DMSWorker_nativeProcess(JNIEnv *env, jclass cls,
    jlong handle, jobject byteBuffer, jint width, jint height, jint stride) {
    // The external data is not automatically de-allocated
    Mat image(height, width, CV_8UC4, env->GetDirectBufferAddress(byteBuffer), stride);
    // Process all face-related stuff
    DMSWorker* worker = reinterpret_cast<DMSWorker*>(handle);
    worker->process(image);
}

