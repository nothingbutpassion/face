package com.hangsheng.face;

import android.graphics.ImageFormat;
import android.graphics.PixelFormat;
import android.media.Image;
import android.view.Surface;

import java.nio.ByteBuffer;

public class NativeBuffer {
    private int mWidth;
    private int mHeight;
    private int mFormat;
    private int mStride;
    private ByteBuffer mByteBuffer;

    public static NativeBuffer from(Image image) {
        // NOTES:
        // Currently, only RGBA and JPEG are supported.
        int width = image.getWidth();
        int height = image.getHeight();
        int format = image.getFormat();
        ByteBuffer byteBuffer = image.getPlanes()[0].getBuffer();
        if (format == PixelFormat.RGBA_8888) {
            int stride = image.getPlanes()[0].getRowStride();
            return new NativeBuffer(byteBuffer, width, height, format, stride);
        }
        if (format == ImageFormat.JPEG) {
            ByteBuffer outBuffer = ByteBuffer.allocateDirect(width*height*4);
            nativeDecode(byteBuffer, byteBuffer.remaining(), outBuffer, width, height, width*4);
            return new NativeBuffer(outBuffer, width, height, PixelFormat.RGBA_8888, width*4);
        }
        return null;
    }


    public NativeBuffer(ByteBuffer byteBuffer, int width, int height, int format, int stride) {
        mByteBuffer = byteBuffer;
        mWidth = width;
        mHeight = height;
        mFormat = format;
        mStride = stride;
    }

    public ByteBuffer getByteBuffer() {
        return mByteBuffer;
    }
    public int getWidth() {
        return  mWidth;
    }
    public int getHeight() {
        return mHeight;
    }
    public int getFormat() {
        return mFormat;
    }
    public int getStride() {
        return mStride;
    }

    public void draw(Surface outoput) {
        if (mFormat == PixelFormat.RGBA_8888) {
            nativeDraw(outoput, mByteBuffer, mWidth, mHeight, mStride);
        }
    }

    // Flipping: 0 - No flipping, 1 - Horizontal flipping  2 - Vertical flipping, 3 - Horizontal & Vertical flipping
    public NativeBuffer flip(int flipCode) {
        if (flipCode == 0) {
            return  this;
        }
//        ByteBuffer outBuffer = ByteBuffer.allocateDirect(mWidth*mHeight*4);
//        nativeFlip(mByteBuffer, mWidth, mHeight, mStride, outBuffer, mWidth*4, flipCode);
//        return  new NativeBuffer(outBuffer, mWidth, mHeight, mFormat, mWidth*4);
        nativeFlip(mByteBuffer, mWidth, mHeight, mStride, mByteBuffer, mStride, flipCode);
        return this;
    }

    // Counter-clockwise rotate degrees: 0, 90, 180, 270
    public NativeBuffer rotate(int rotateCode) {
        if (rotateCode == 0) {
            // NEED NO rotating
            return  this;
        }
        ByteBuffer outBuffer = ByteBuffer.allocateDirect(mWidth*mHeight*4);
        if (rotateCode == 90 || rotateCode == 270) {
            nativeRotate(mByteBuffer, mWidth, mHeight, mStride, outBuffer, mHeight*4, rotateCode);
            return new NativeBuffer(outBuffer, mHeight, mWidth, mFormat, mHeight*4);
        } else {
            nativeRotate(mByteBuffer, mWidth, mHeight, mStride, outBuffer, mWidth*4, rotateCode);
            return  new NativeBuffer(outBuffer, mWidth, mHeight, mFormat, mWidth*4);
        }
    }

    private static native void nativeDraw(Surface outoput, ByteBuffer byteBuffer, int width, int height, int stride);
    private static native void nativeFlip(ByteBuffer srcBuffer, int srcWidth, int srcHeight, int srcStride,
                                          ByteBuffer dstBuffer, int dstStride, int flipCode);
    private static native void nativeRotate(ByteBuffer srcBuffer, int srcWidth, int srcHeight, int srcStride,
                                          ByteBuffer dstBuffer, int dstStride, int rotateCode);

    private static native void nativeDecode(ByteBuffer srcBuffer, int srcSize, ByteBuffer dstBuffer, int dstWidth, int dstHeight, int dstStride);
}
