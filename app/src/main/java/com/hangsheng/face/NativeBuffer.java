package com.hangsheng.face;

import android.graphics.PixelFormat;
import android.media.Image;
import android.view.Surface;

import java.nio.ByteBuffer;

public class NativeBuffer {
    private int mWidth;
    private int mHeight;
    private int mFormat;
    private int mStride;
    private int mOffset;
    private ByteBuffer mByteBuffer;

    public NativeBuffer(Image image) {
        mWidth =image.getWidth();
        mHeight = image.getHeight();
        mFormat = image.getFormat();
        mStride = image.getPlanes()[0].getRowStride();
        mByteBuffer = image.getPlanes()[0].getBuffer();
        mOffset = 0;
    }

    public NativeBuffer(ByteBuffer byteBuffer, int width, int height, int format, int stride, int offset) {
        mByteBuffer = byteBuffer;
        mWidth = width;
        mHeight = height;
        mFormat = format;
        mStride = stride;
        mOffset = offset;
    }

    public NativeBuffer(ByteBuffer byteBuffer, int width, int height, int format, int stride) {
        this(byteBuffer, width, height, format, stride, 0);
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
    public int getOffset() {
        return mOffset;
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
            return  new NativeBuffer(outBuffer, mHeight, mWidth, mFormat, mHeight*4);
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
}
