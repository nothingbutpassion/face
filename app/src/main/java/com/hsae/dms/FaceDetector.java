package com.hsae.dms;

import android.graphics.PixelFormat;
import android.graphics.PointF;
import android.graphics.Rect;
import android.os.Environment;

import java.nio.ByteBuffer;

public class FaceDetector {

    // Native face detector handle
    private long mNativeHandle = 0;


    public boolean open() {
        mNativeHandle = nativeCreate(Environment.getExternalStorageDirectory() + "/Face");
        return (mNativeHandle != 0);
    }

    public void close() {
        if (mNativeHandle != 0) {
            nativeDestroy(mNativeHandle);
        }
    }

    public boolean process(NativeBuffer nativeBuffer) {
        if (mNativeHandle == 0 || nativeBuffer.getFormat() != PixelFormat.RGBA_8888) {
            return false;
        }
       nativeProcess(mNativeHandle,  nativeBuffer.getByteBuffer(),
               nativeBuffer.getWidth(), nativeBuffer.getHeight(), nativeBuffer.getStride());
        return true;
    }

    public Rect[] findFaces(NativeBuffer nativeBuffer) {
        Rect[] objectRects = new Rect[0];
        if (mNativeHandle != 0 && nativeBuffer.getFormat() == PixelFormat.RGBA_8888) {
            objectRects = nativeDetect(mNativeHandle,  nativeBuffer.getByteBuffer(),
                    nativeBuffer.getWidth(), nativeBuffer.getHeight(), nativeBuffer.getStride());
        }
        return objectRects;
    }

    public PointF[] getMarks(NativeBuffer nativeBuffer, Rect face) {
        PointF[] marks = new PointF[0];
        if (mNativeHandle != 0 && nativeBuffer.getFormat() == PixelFormat.RGBA_8888) {
            marks = nativeGetMarks(mNativeHandle, nativeBuffer.getByteBuffer(),
                    nativeBuffer.getWidth(), nativeBuffer.getHeight(), nativeBuffer.getStride(), face);
        }
        return marks;
    }

    // Create native face detector
    private static native long nativeCreate(String modelDir) ;

    // Destroy native face detector
    private static native void nativeDestroy(long nativeHandle);

    // Face detection for RGBA_8888 image
    private static native Rect[] nativeDetect(long nativeHandle, ByteBuffer byteBuffer, int width, int height, int stride);

    // Face marks for RGBA_8888 image
    private static native PointF[] nativeGetMarks(long nativeHandle, ByteBuffer byteBuffer, int width, int height, int stride, Rect roi);

    // Process all face related stuff
    private static native void nativeProcess(long nativeHandle, ByteBuffer byteBuffer, int width, int height, int stride);

}
