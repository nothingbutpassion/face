#ifndef NATIVEBUFFER_JNI_H
#define NATIVEBUFFER_JNI_H

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif
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


JNIEXPORT void JNICALL Java_com_hsae_dms_NativeBuffer_native420ToRGBA(JNIEnv* env, jclass cls,
                                                                       jobject y, jobject u, jobject v, int uStride,
                                                                       jint width, jint height, jobject dstBuffer);

JNIEXPORT void JNICALL Java_com_hsae_dms_NativeBuffer_nativeNV21ToRGBA(JNIEnv* env, jclass cls,
                                                                       jbyteArray srcBuffer, jobject dstBuffer,
                                                                       jint dstWidth, jint dstHeight, jint dstStride);
#ifdef __cplusplus
}
#endif

#endif // NATIVEBUFFER_JNI_H
