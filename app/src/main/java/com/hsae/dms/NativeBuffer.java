package com.hsae.dms;

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

    /**
     * Create NativeBuffer from an Image
     * @param image a JPEG or RGBA Image
     * @return A NativeBuffer if succeed, otherwise return null
     */
    public static NativeBuffer fromImage(Image image) {
        int width = image.getWidth();
        int height = image.getHeight();
        int format = image.getFormat();
        ByteBuffer byteBuffer = image.getPlanes()[0].getBuffer();
        if (format == PixelFormat.RGBA_8888) {
            int stride = image.getPlanes()[0].getRowStride();
            return new NativeBuffer(byteBuffer, width, height, format, stride);
        } else if (format == ImageFormat.JPEG) {
            ByteBuffer outBuffer = ByteBuffer.allocateDirect(width*height*4);
            nativeDecode(byteBuffer, byteBuffer.remaining(), outBuffer, width, height, width*4);
            return new NativeBuffer(outBuffer, width, height, PixelFormat.RGBA_8888, width*4);
        } else if (format == ImageFormat.YUV_420_888) {
            Image.Plane[] planes = image.getPlanes();
            ByteBuffer y = planes[0].getBuffer();
            ByteBuffer u = planes[1].getBuffer();
            ByteBuffer v = planes[2].getBuffer();
            int uStride = planes[1].getPixelStride();
            // NOTES:
            // For ImageFormat.YUV_420_888, u has same stride with v
            ByteBuffer outBuffer = ByteBuffer.allocateDirect(width*height*4);
            native420ToRGBA(y, u, v, uStride, width, height, outBuffer);
            return new NativeBuffer(outBuffer, width, height, PixelFormat.RGBA_8888, width*4);
        }
        return null;
    }

    /**
     * @brief Create NativeBuffer from an Image
     * @param nv21 image buffer
     * @param width image width
     * @param height image height
     * @return A NativeBuffer created based on specified format/width/height
     */
    public static NativeBuffer fromNV21(byte[] nv21, int width, int height) {
            ByteBuffer rgbaBuffer = ByteBuffer.allocateDirect(width*height*4);
            nativeNV21ToRGBA(nv21, rgbaBuffer, width, height, width*4);
            return new NativeBuffer(rgbaBuffer, width, height, PixelFormat.RGBA_8888, width*4);
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

    /**
     * Flip NativeBuffer with specified method
     * @param flipCode flip method: 0 - No flipping, 1 - Horizontal flipping  2 - Vertical flipping, 3 - Horizontal & Vertical flipping
     * @return A flipped NativeBuffer
     */
    public NativeBuffer flip(int flipCode) {
        if (flipCode == 0) {
            return  this;
        }
        nativeFlip(mByteBuffer, mWidth, mHeight, mStride, mByteBuffer, mStride, flipCode);
        return this;
    }

    /**
     * Rotate NativeBuffer in counter-clockwise way
     * @param rotateCode Specify rotate degrees, valid value are:  0, 90, 180, 270
     * @return A rotated NativeBuffer
     */
    public NativeBuffer rotate(int rotateCode) {
        if (rotateCode == 0) {
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

    private static native void nativeDecode(ByteBuffer srcBuffer, int srcSize,
                                            ByteBuffer dstBuffer, int dstWidth, int dstHeight, int dstStride);
    private static native void native420ToRGBA(ByteBuffer y, ByteBuffer u, ByteBuffer v, int uStride,
                                               int width, int height, ByteBuffer dstBuffer);
    private static native void nativeNV21ToRGBA(byte[] srcBuffer, ByteBuffer dstBuffer,
                                                int dstWidth, int dstHeight, int dstStride);


}
