package com.hangsheng.face;

import android.app.Service;
import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.ImageFormat;
import android.graphics.PixelFormat;
import android.graphics.Point;
import android.graphics.PointF;
import android.graphics.Rect;
import android.media.Image;
import android.os.Environment;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.util.logging.Handler;

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

    public Rect[] detect(Image image) {
        Rect[] objectRects = new Rect[0];
        if (mNativeHandle != 0 && image.getFormat() == PixelFormat.RGBA_8888) {
            Image.Plane plane = image.getPlanes()[0];
            ByteBuffer byteBuffer = plane.getBuffer();
            int stride = plane.getRowStride();
            int width = image.getWidth();
            int height = image.getHeight();
            objectRects = nativeDetect(mNativeHandle, byteBuffer, width, height, stride);
        }
        return objectRects;
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

}
