#include "native_buffer_jni.h"
#include <android/native_window_jni.h>
#include <android/native_window.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "utils.h"

using namespace cv;

//
// com.hsae.dms.NativeBuffer
//
JNIEXPORT void JNICALL Java_com_hsae_dms_NativeBuffer_nativeDraw(JNIEnv* env, jclass cls,
    jobject surface, jobject byteBuffer, jint width, jint height, jint stride) {
    ANativeWindow* window = ANativeWindow_fromSurface(env, surface);
    if (window) {
        ANativeWindow_Buffer buffer = {0};
        if (ANativeWindow_lock(window, &buffer, 0) == 0) {
            assert(buffer.format == WINDOW_FORMAT_RGBA_8888);
            Mat display(buffer.height, buffer.width, CV_8UC4, buffer.bits, buffer.stride*4);
            Mat image(height, width, CV_8UC4, env->GetDirectBufferAddress(byteBuffer), stride);
            Mat center;
            float ration;
            if (buffer.width*height > buffer.height*width) {
                ration = buffer.height/float(height);
                int w = int(ration * width);
                int h = buffer.height;
                Rect roi((buffer.width-w)/2, 0, w, h);
                center = display(roi);
            } else {
                ration = buffer.width/float(width);
                int h = int(ration * height);
                int w = buffer.width;
                Rect roi(0, (buffer.height-h)/2, w, h);
                center = display(roi);
            }
            resize(image, center, center.size());
            ANativeWindow_unlockAndPost(window);
        }
        ANativeWindow_release(window);
    }
}
JNIEXPORT void JNICALL Java_com_hsae_dms_NativeBuffer_nativeFlip(JNIEnv* env, jclass cls,
    jobject srcBuffer, jint srcWidth, jint srcHeight, jint srcStride, jobject dstBuffer, jint dstStride, jint flipCode) {
    Mat src(srcHeight, srcWidth, CV_8UC4, env->GetDirectBufferAddress(srcBuffer), srcStride);
    Mat dst(srcHeight, srcWidth, CV_8UC4, env->GetDirectBufferAddress(dstBuffer), dstStride);
    // flipCode: 0 - No flipping, 1 - Horizontal flipping  2 - Vertical flipping, 3 - Horizontal & Vertical flipping
    if (flipCode == 1) {
        // OpenCV flip: Horizontal flipping if flipCode > 0
        flip(src, dst, 1);
    } else if (flipCode == 2) {
        // OpenCV flip: Vertical flipping if flipCode == 0
        flip(src, dst, 0);
    } else if (flipCode == 3) {
        // OpenCV flip: Horizontal flipping if flipCode < 0
        flip(src, dst, -1);
    } else {
        // No flipping, just copying
        src.copyTo(dst);
    }
}
JNIEXPORT void JNICALL Java_com_hsae_dms_NativeBuffer_nativeRotate(JNIEnv* env, jclass cls,
    jobject srcBuffer, jint srcWidth, jint srcHeight, jint srcStride, jobject dstBuffer, jint dstStride, jint rotateCode) {
    Mat src(srcHeight, srcWidth, CV_8UC4, env->GetDirectBufferAddress(srcBuffer), srcStride);
    // rotateCode - counterclockwise rotate degrees: 0, 90, 180, 270
    if (rotateCode == 90) {
        Mat dst(srcWidth, srcHeight, CV_8UC4, env->GetDirectBufferAddress(dstBuffer), dstStride);
        rotate(src, dst, ROTATE_90_CLOCKWISE);
    } else if (rotateCode == 180) {
        Mat dst(srcHeight, srcWidth, CV_8UC4, env->GetDirectBufferAddress(dstBuffer), dstStride);
        rotate(src, dst, ROTATE_180);
    } else if (rotateCode == 270) {
        Mat dst(srcWidth, srcHeight, CV_8UC4, env->GetDirectBufferAddress(dstBuffer), dstStride);
        rotate(src, dst, ROTATE_90_COUNTERCLOCKWISE);
    } else {
        // no rotating, just copying
        Mat dst(srcHeight, srcWidth, CV_8UC4, env->GetDirectBufferAddress(dstBuffer), dstStride);
        src.copyTo(dst);
    }
}

JNIEXPORT void JNICALL Java_com_hsae_dms_NativeBuffer_nativeDecode(JNIEnv* env, jclass cls,
    jobject srcBuffer, jint srcSize, jobject dstBuffer, jint dstWidth, jint dstHeight, jint dstStride) {
    Mat src(1, srcSize, CV_8UC1, env->GetDirectBufferAddress(srcBuffer), srcSize);
    Mat dst(dstHeight, dstWidth, CV_8UC4, env->GetDirectBufferAddress(dstBuffer), dstStride);
    Mat img = imdecode(src, IMREAD_COLOR);
    cvtColor(img, dst, COLOR_BGR2RGBA);
}

JNIEXPORT void JNICALL Java_com_hsae_dms_NativeBuffer_nativeNV21ToRGBA(JNIEnv* env, jclass cls,
    jbyteArray srcBuffer, jobject dstBuffer, jint dstWidth, jint dstHeight, jint dstStride) {
    Mat rgba(dstHeight, dstWidth, CV_8UC4, env->GetDirectBufferAddress(dstBuffer), dstStride);
    void* src = env->GetPrimitiveArrayCritical(srcBuffer, 0);
    Mat nv21(dstHeight + dstHeight/2, dstWidth, CV_8UC1, src);
    cvtColor(nv21, rgba, COLOR_YUV2RGBA_NV21);
    env->ReleasePrimitiveArrayCritical(srcBuffer, src, JNI_ABORT);
}


