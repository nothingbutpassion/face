package com.hangsheng.face;

import android.graphics.ImageFormat;
import android.hardware.Camera;
import android.util.Log;
import android.view.SurfaceHolder;

import java.io.IOException;
import java.nio.ByteBuffer;

public class DeprecatedCapture {

    private final static String TAG = "DeprecatedCapture";
    private static Camera mCamera;

    // This listener is invoked when an image is captured
    public interface CaptureListener {
        void onCaptured(byte[] nv21, int width, int height);
    }

    public static boolean open(SurfaceHolder holder, CaptureListener listener) {
        if (Camera.getNumberOfCameras() < 1) {
            Log.e(TAG, "no cameras found");
            return false;
        }
        Camera camera = Camera.open();
        if (camera == null) {
            Log.w(TAG, "no back camera found");
            camera = Camera.open(0);
        }

        try {
            camera.setPreviewDisplay(holder);
        } catch (IOException e) {
            e.printStackTrace();
            camera.release();
            return false;
        }

        final CaptureListener captureListener = listener;
        Camera.Parameters parameters = camera.getParameters();
        final Camera.Size size = parameters.getSupportedPreviewSizes().get(0);
        parameters.setPreviewSize(size.width, size.height);
        parameters.setPreviewFormat(ImageFormat.NV21);
        camera.setParameters(parameters);
        ByteBuffer directBuffer = ByteBuffer.allocateDirect(size.width * size.height * ImageFormat.getBitsPerPixel(ImageFormat.NV21) / 8);
        camera.addCallbackBuffer(directBuffer.array());
        camera.setPreviewCallbackWithBuffer(new Camera.PreviewCallback() {
            @Override
            public void onPreviewFrame(byte[] data, Camera camera) {
                camera.addCallbackBuffer(ByteBuffer.allocateDirect(data.length).array());
                if (captureListener != null) {
                    captureListener.onCaptured(data, size.width, size.height);
                }
            }
        });
        camera.startPreview();
        return true;
    }

    public static void close() {
        if (mCamera != null) {
            mCamera.stopPreview();
            mCamera.release();
            mCamera = null;
        }
    }
}
