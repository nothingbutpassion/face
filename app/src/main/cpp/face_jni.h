#ifndef FACE_FACE_JNI_H
#define FACE_FACE_JNI_H

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif

//
// com.hsae.dms.FaceDetector
//
JNIEXPORT jlong JNICALL Java_com_hsae_dms_FaceDetector_nativeCreate(JNIEnv* env, jclass cls, jstring modelDir);
JNIEXPORT void JNICALL Java_com_hsae_dms_FaceDetector_nativeDestroy(JNIEnv* env, jclass cls, jlong handle);
JNIEXPORT jobjectArray JNICALL Java_com_hsae_dms_FaceDetector_nativeDetect(JNIEnv *env, jclass cls,
    jlong handle, jobject byteBuffer, jint width, jint height, jint stride);
JNIEXPORT jobjectArray JNICALL Java_com_hsae_dms_FaceDetector_nativeGetMarks(JNIEnv *env, jclass cls,
    jlong handle, jobject byteBuffer, jint width, jint height, jint stride, jobject roi);
// Process all face-related stuff
JNIEXPORT void JNICALL Java_com_hsae_dms_FaceDetector_nativeProcess(JNIEnv *env, jclass cls,
    jlong handle, jobject byteBuffer, jint width, jint height, jint stride);
//
// com.hsae.dms.NativeBuffer
//
JNIEXPORT void JNICALL Java_com_hsae_dms_NativeBuffer_nativeDraw(JNIEnv* env, jclass cls,
    jobject surface, jobject byteBuffer, jint width, jint height, jint stride);
JNIEXPORT void JNICALL Java_com_hsae_dms_NativeBuffer_nativeFlip(JNIEnv* env, jclass cls,
    jobject srcBuffer, jint srcWidth, jint srcHeight, jint srcStride, jobject dstBuffer, jint dstStride, jint flipCode);
JNIEXPORT void JNICALL Java_com_hsae_dms_NativeBuffer_nativeRotate(JNIEnv* env, jclass cls,
    jobject srcBuffer, jint srcWidth, jint srcHeight, jint srcStride, jobject dstBuffer, jint dstStride, jint rotateCode);
JNIEXPORT void JNICALL Java_com_hsae_dms_NativeBuffer_nativeDecode(JNIEnv* env, jclass cls,
    jobject srcBuffer, jint srcSize, jobject dstBuffer, jint dstWidth, jint dstHeight, jint dstStride);
JNIEXPORT void JNICALL Java_com_hsae_dms_NativeBuffer_nativeNV21ToRGBA(JNIEnv* env, jclass cls,
    jbyteArray srcBuffer, jobject dstBuffer, jint dstWidth, jint dstHeight, jint dstStride);
#ifdef __cplusplus
}
#endif

#endif //FACE_FACE_JNI_H
