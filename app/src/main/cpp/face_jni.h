#ifndef FACE_FACE_JNI_H
#define FACE_FACE_JNI_H

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif

//
// com.hangsheng,face.FaceDetector
//
JNIEXPORT jlong JNICALL Java_com_hangsheng_face_FaceDetector_nativeCreate(JNIEnv* env, jclass cls, jstring modelDir);
JNIEXPORT void JNICALL Java_com_hangsheng_face_FaceDetector_nativeDestroy(JNIEnv* env, jclass cls, jlong handle);
JNIEXPORT jobjectArray JNICALL Java_com_hangsheng_face_FaceDetector_nativeDetect(JNIEnv *env, jclass cls,
    jlong handle, jobject byteBuffer, jint width, jint height, jint stride);
JNIEXPORT jobjectArray JNICALL Java_com_hangsheng_face_FaceDetector_nativeGetMarks(JNIEnv *env, jclass cls,
    jlong handle, jobject byteBuffer, jint width, jint height, jint stride, jobject roi);
// Process all face-related stuff
JNIEXPORT void JNICALL Java_com_hangsheng_face_FaceDetector_nativeProcess(JNIEnv *env, jclass cls,
    jlong handle, jobject byteBuffer, jint width, jint height, jint stride);
//
// com.hangsheng,face.NativeBuffer
//
JNIEXPORT void JNICALL Java_com_hangsheng_face_NativeBuffer_nativeDraw(JNIEnv* env, jclass cls,
    jobject surface, jobject byteBuffer, jint width, jint height, jint stride);
JNIEXPORT void JNICALL Java_com_hangsheng_face_NativeBuffer_nativeFlip(JNIEnv* env, jclass cls,
    jobject srcBuffer, jint srcWidth, jint srcHeight, jint srcStride, jobject dstBuffer, jint dstStride, jint flipCode);
JNIEXPORT void JNICALL Java_com_hangsheng_face_NativeBuffer_nativeRotate(JNIEnv* env, jclass cls,
    jobject srcBuffer, jint srcWidth, jint srcHeight, jint srcStride, jobject dstBuffer, jint dstStride, jint rotateCode);
#ifdef __cplusplus
}
#endif

#endif //FACE_FACE_JNI_H
