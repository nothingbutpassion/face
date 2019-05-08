#ifndef DMSWORKER_JNI_H
#define DMSWORKER_JNI_H

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif

//
// com.hsae.dms.ImageProcessor
//
JNIEXPORT jlong JNICALL Java_com_hsae_dms_ImageProcessor_nativeCreate(JNIEnv* env, jclass cls, jstring modelDir);
JNIEXPORT void JNICALL Java_com_hsae_dms_ImageProcessor_nativeDestroy(JNIEnv* env, jclass cls, jlong handle);
// Process all face-related stuff
JNIEXPORT void JNICALL Java_com_hsae_dms_ImageProcessor_nativeProcess(JNIEnv *env, jclass cls,
    jlong handle, jobject byteBuffer, jint width, jint height, jint stride);

#ifdef __cplusplus
}
#endif

#endif // DMSWORKER_JNI_H
