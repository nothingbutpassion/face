package com.hsae.dms;
import android.graphics.PixelFormat;
import android.os.Environment;
import java.nio.ByteBuffer;

public class ImageProcessor {
    private long mNativeHandle = 0;

    public boolean open() {
        mNativeHandle = nativeCreate(Environment.getExternalStorageDirectory() + "/dms");
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

    // Create native image processor
    private static native long nativeCreate(String modelDir) ;

    // Destroy native image processor
    private static native void nativeDestroy(long nativeHandle);

    // Native image process
    private static native void nativeProcess(long nativeHandle, ByteBuffer byteBuffer, int width, int height, int stride);

}
