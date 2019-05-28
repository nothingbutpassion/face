package com.hsae.dms;

import android.graphics.ImageFormat;
import android.graphics.SurfaceTexture;
import android.hardware.Camera;
import android.util.Log;


import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.List;

public class CameraUtils {
    private final static String TAG = "CameraUtils";
    private Camera mCamera;
    private CaptureListener mCaptureListener;
    private int mCaptureFps = 15;
    private int mCaptureWidth = 640;
    private int mCaptureHeight = 480;

    // This listener is invoked when an image is captured
    public interface CaptureListener {
        void onCaptured(byte[] nv21, int width, int height);
    }

    public boolean startPreview(SurfaceTexture surfaceTexture) {
        if (mCamera == null) {
            Log.e(TAG, "no camera opened");
            return false;
        }
        try {
            mCamera.setPreviewTexture(surfaceTexture);
        } catch (IOException e) {
            e.printStackTrace();
            mCamera.release();
            return false;
        }
        mCamera.startPreview();
        return true;
    }

    public void setCaptureListener(CaptureListener listener) {
        mCaptureListener = listener;
    }

    public void setCaptureSize(int width, int height) {
        if (width > 0 && height > 0) {
            mCaptureWidth = width;
            mCaptureHeight = height;
        }
    }

    public void setCaptureFps(int fps) {
        if (fps > 0) {
            mCaptureFps = fps;
        }
    }

//    public int getCaptureWidth() { return mCaptureWidth; }
//    public int getCaptureHeight() { return mCaptureHeight; }
//    public int getCaptureFps() { return mCaptureFps; }

    private Camera.Size selectCaptureSize(Camera.Parameters parameters) {
        List<Camera.Size> sizes = parameters.getSupportedPictureSizes();
        Camera.Size size = sizes.get(0);
        float maxIOU = (float)(Math.min(size.width, mCaptureWidth)*Math.min(size.height, mCaptureHeight))
                /(Math.max(size.width, mCaptureWidth)*Math.max(size.height, mCaptureHeight));
        for (Camera.Size s: sizes) {
            Log.i(TAG,"supported size: " + s.width + "x"  + s.height);
            float iou = (float)(Math.min(s.width, mCaptureWidth)*Math.min(s.height, mCaptureHeight))
                    /(Math.max(s.width, mCaptureWidth)*Math.max(s.height, mCaptureHeight));
            if (iou > maxIOU) {
                maxIOU = iou;
                size = s;
            }
        }
        Log.i(TAG, "selected size:"  + size.width + "x"  + size.height);
        return size;
    }

    private int[] selectFps(Camera.Parameters parameters) {
        List<int[]> fpsRanges = parameters.getSupportedPreviewFpsRange();
        int[] fpsRange = fpsRanges.get(0);
        for (int[] range: fpsRanges) {
            Log.i(TAG,"supported fps range: [" + range[0]/1000.0 + ", " + range[1]/1000.0 + "]");
            if  (range[0] <= 1000*mCaptureFps &&  1000*mCaptureFps <= range[1]) {
                fpsRange = range;
                // Try to set the specified fps
                parameters.setPreviewFrameRate(mCaptureFps);
                break;
            }
        }
        Log.i(TAG, "selected fps range: [" + fpsRange[0]/1000.0 + ", " + fpsRange[1]/1000.0 + "]");
        return fpsRange;
    }

    public boolean open() {
        close();
        int numCameras = Camera.getNumberOfCameras();
        if (numCameras < 1) {
            Log.e(TAG, "no cameras found");
            return false;
        }
        Camera.CameraInfo cameraInfo = new Camera.CameraInfo();
        for (int cameraIndex = 0; cameraIndex < numCameras; ++cameraIndex) {
            Camera.getCameraInfo(cameraIndex, cameraInfo);
            if (cameraInfo.facing == Camera.CameraInfo.CAMERA_FACING_FRONT) {
                mCamera = Camera.open(cameraIndex);
                break;
            }
        }
        if (mCamera == null)
            mCamera = Camera.open(0);
        Camera.Parameters parameters = mCamera.getParameters();
        int[] fpsRange = selectFps(parameters);
        final Camera.Size size = selectCaptureSize(parameters);
        parameters.setPreviewSize(size.width, size.height);
        parameters.setPreviewFpsRange(fpsRange[0], fpsRange[1]);
        parameters.setPreviewFormat(ImageFormat.NV21);
        mCamera.setParameters(parameters);
        mCamera.addCallbackBuffer(ByteBuffer.allocateDirect(size.width * size.height * 3/2).array());
        mCamera.addCallbackBuffer(ByteBuffer.allocateDirect(size.width * size.height * 3/2).array());
        mCamera.setPreviewCallbackWithBuffer(new Camera.PreviewCallback() {
            @Override
            public void onPreviewFrame(byte[] data, Camera camera) {
                camera.addCallbackBuffer(data.clone());
                if (mCaptureListener != null) {
                    mCaptureListener.onCaptured(data, size.width, size.height);
                }
            }
        });
        mCaptureWidth = parameters.getPreviewSize().width;
        mCaptureHeight = parameters.getPreviewSize().height;
        mCaptureFps = parameters.getPreviewFrameRate();
        Log.i(TAG,"capture params: format="  + parameters.getPreviewFormat()
                + ", width=" + mCaptureWidth + ", height=" + mCaptureHeight + ", fps=" + mCaptureFps);
        return true;
    }

    public void close() {
        if (mCamera != null) {
            mCamera.stopPreview();
            mCamera.release();
            mCamera = null;
        }
    }
}
