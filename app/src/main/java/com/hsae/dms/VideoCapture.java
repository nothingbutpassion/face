package com.hsae.dms;

import android.app.Service;
import android.content.Context;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.PixelFormat;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.ImageReader;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;
import android.util.Size;
import android.view.Surface;
import android.view.WindowManager;

import java.util.ArrayList;
import java.util.List;


public class VideoCapture {
    private static final String TAG = "VideoCapture";

    private final Context mContext;

    // These params can be set before open, may be changed in open, but fixed after open
    private int mCameraFacing = CameraCharacteristics.LENS_FACING_FRONT;
    private int mOutputWidth = 1280;
    private int mOutputHeight = 720;
    private int mOutputFormat = PixelFormat.RGBA_8888;
    private Surface mPreviewSurface;

    // Capture listener can be set before open, can be changed after open
    // NOTES: current implementation is not thread-safe
    private CaptureListener mCaptureListener;

    // Camera sensor-orientation/device/capture-session is orderly got in open process..
    private int mSensorOrientation = 0;
    private CameraDevice mCameraDevice;
    private CameraCaptureSession mCameraCaptureSession;

    // Background thread & handler
    private HandlerThread mBackgroundThread;
    private Handler mBackgroundHandler;

    // This listener is invoked when an image is captured
    public interface CaptureListener {
        void onCaptured(Image image);
    }

    public VideoCapture(Context context) {
        mContext = context;
    }

    public void setPreviewSurface(Surface surface) {
        mPreviewSurface = surface;
    }

    // NOTES:
    // Camera device: 0 - front camera, 1 - back camera, 2 - external camera.
    // Must be same as CameraCharacteristics.LENS_FACING_FRONT/BACK/EXTERNAL
    public void setCaptureDevice(int device) {
        mCameraFacing = device;
    }

    public void setCaptureImage(int width, int height, int format) {
        mOutputWidth = width;
        mOutputHeight = height;
        mOutputFormat = format;
    }

    public void setCaptureListener(CaptureListener captureListener) {
        mCaptureListener = captureListener;
    }

    public boolean open() {
        final String cameraId = selectCamera();
        if (cameraId == null) {
            Log.e(TAG, "no camera found");
            return false;
        }
        CameraManager manager = (CameraManager) mContext.getSystemService(Service.CAMERA_SERVICE);
        try {
            startBackgroundThread();
            manager.openCamera(cameraId, new CameraDevice.StateCallback() {
                @Override
                public void onOpened(CameraDevice camera) {
                    mCameraDevice = camera;
                    Log.i(TAG, "camera: " + cameraId + " opened");
                    startCapture();
                }
                @Override
                public void onDisconnected(CameraDevice camera) {
                    Log.e(TAG, "camera: " + cameraId + " disconnected");
                    camera.close();
                    mCameraDevice = null;
                }
                @Override
                public void onError(CameraDevice camera, int error) {
                    Log.e(TAG, "camera: " + cameraId + " error: " + error);
                    camera.close();
                    mCameraDevice = null;
                }
            }, mBackgroundHandler);
        } catch (SecurityException | CameraAccessException e) {
            e.printStackTrace();
            stopBackgroundThread();
            return false;
        }
        return true;
    }

    public void close() {
        abortCapture();
        if (mCameraDevice != null) {
            mCameraDevice.close();
        }
        stopBackgroundThread();
    }

    public Size getCaptureSize() {
        return new Size(mOutputWidth, mOutputHeight);
    }

    // Counter-clockwise rotate degrees: 0, 90, 180, 270
    public int getCaptureRotation() {
        // NOTES:
        // 1) Output image needs to be rotated clockwise to be upright on the device screen in its native orientation.
        // 2) But Display is rotated counter-clockwise from its "natural" orientation
        // The total rotation of output image should be calculated as following formula.
       return (mSensorOrientation - getDisplayRotation() + 360)%360;
    }

    // Flipping: 0 - No flipping, 1 - Horizontal flipping  2 - Vertical flipping, 3 - Horizontal & Vertical flipping
    public int getCaptureFlipping() {
        // 0 - No flipping
        if (mCameraFacing != CameraCharacteristics.LENS_FACING_FRONT) {
            return 0;
        }
        // 1 - Horizontal flipping
        int rotation = getCaptureRotation();
        if (rotation == 90 || rotation == 270) {
            return 1;
        }
        // 2 - Vertical flipping
        return 2;
    }

    public Matrix getCaptureTransform(int imageWidth, int imageHeight, int displayWidth, int displayHeight) {
        Matrix matrix = new Matrix();

        // NOTES:
        // 1) Output image needs to be rotated clockwise to be upright on the device screen in its native orientation.
        // 2) But Display is rotated counter-clockwise from its "natural" orientation
        // The total rotation of output image should be calculated as following formula.
        int totalRotation = (mSensorOrientation - getDisplayRotation() + 360)%360;

        // First rotate the image for displaying image
        matrix.postRotate(totalRotation, imageWidth/2.0f, imageHeight/2.0f);

        // The image width and height may be swapped after rotation
        boolean swappedDimensions = (totalRotation == 90 || totalRotation == 270);
        int width = swappedDimensions ? imageHeight : imageWidth;
        int height = swappedDimensions ? imageWidth : imageHeight;

        // NOTES:
        // For front camera, MUST do Mirror Transformation based on whether image width and height is swapped after rotation.
        if  (mCameraFacing == CameraCharacteristics.LENS_FACING_FRONT) {
            matrix.postTranslate(-imageWidth/2.0f, -imageHeight/2.0f);
            if (swappedDimensions) {
                matrix.postScale(-1.0f, 1.0f);
            } else {
                matrix.postScale(1.0f, -1.0f);
            }
            matrix.postTranslate(imageWidth/2.0f, imageHeight/2.0f);
        }

        // Move image center to display center
        matrix.postTranslate((displayWidth-imageWidth)/2.0f, (displayHeight - imageHeight)/2.0f);

        // Scale image to fit screen display rectangle
        float ration = (float)displayHeight/(float)height;
        if (width*ration > displayWidth) {
            ration = (float)displayWidth/(float)width;
        }
        matrix.postScale(ration, ration, displayWidth/2.0f, displayHeight/2.0f);

        return  matrix;
    }

    // The rotation of the screen from its "natural" orientation: 0, 90, 180, 270 degrees.
    // For example, if the device is rotated 90 degrees counter-clockwise, to compensate rendering will
    // be rotated by 90 degrees clockwise and thus the returned value here will be Surface.ROTATION_90.
    private int getDisplayRotation() {
        WindowManager windowManager = (WindowManager) mContext.getSystemService(Service.WINDOW_SERVICE);
        switch (windowManager.getDefaultDisplay().getRotation()) {
            case Surface.ROTATION_0:
                return 0;
            case Surface.ROTATION_90:
                return 90;
            case Surface.ROTATION_180:
                return 180;
            default:
                return 270;
        }
    }

    private void startCapture() {
        try {
            // Setup ImageReader to read image when captured
            final ImageReader imageReader = ImageReader.newInstance(mOutputWidth, mOutputHeight, mOutputFormat, 4);
            imageReader.setOnImageAvailableListener(new ImageReader.OnImageAvailableListener() {
                @Override
                public void onImageAvailable(ImageReader reader) {
                    Image image = reader.acquireNextImage();
                    if (image != null) {
                        if (mCaptureListener != null) {
                            mCaptureListener.onCaptured(image);
                        }
                        image.close();
                    }
                }
            }, mBackgroundHandler);
            // NOTES:
            // Arrays.asList() returns a fixed-sized list which can be changed.
            List<Surface> outputSurfaces = new ArrayList<>();
            outputSurfaces.add(imageReader.getSurface());
            if (mPreviewSurface != null) {
                outputSurfaces.add(mPreviewSurface);
            }

            // Create capture session
            mCameraDevice.createCaptureSession(outputSurfaces, new CameraCaptureSession.StateCallback() {
                @Override
                public void onConfigured(CameraCaptureSession session) {
                    try {
                        mCameraCaptureSession = session;
                        // Create capture request
                        // NOTES:
                        // Template type CameraDevice.TEMPLATE_PREVIEW can't work on imx8 device
                        int templateType = (mOutputFormat == PixelFormat.RGBA_8888 ? CameraDevice.TEMPLATE_PREVIEW : CameraDevice.TEMPLATE_STILL_CAPTURE);
                        CaptureRequest.Builder builder = mCameraDevice.createCaptureRequest(templateType);
                        builder.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE);
                        builder.addTarget(imageReader.getSurface());
                        if (mPreviewSurface != null) {
                            builder.addTarget(mPreviewSurface);
                        }
                        // Send repeating capture request
                        mCameraCaptureSession.setRepeatingRequest(builder.build(), null, mBackgroundHandler);
                    } catch (CameraAccessException e) {
                        e.printStackTrace();
                    }
                }
                @Override
                public void onConfigureFailed(CameraCaptureSession session) {
                    mCameraCaptureSession = null;
                }
            }, mBackgroundHandler);

        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }

    private void abortCapture() {
        if (mCameraCaptureSession != null) {
            try {
                mCameraCaptureSession.stopRepeating();
                mCameraCaptureSession.abortCaptures();
                mCameraCaptureSession.close();
            } catch (CameraAccessException e) {
                e.printStackTrace();
            }
        }
    }

    private String selectCamera() {
        String selected = null;
        CameraManager manager = (CameraManager) mContext.getSystemService(Service.CAMERA_SERVICE);
        try {
            String[] cameraIds = manager.getCameraIdList();
            if (cameraIds == null || cameraIds.length == 0) {
                return selected;
            }
            // Select the camera which meets facing requirement.
            CameraCharacteristics characteristics = null;
            for (final String cameraId : cameraIds) {
                characteristics = manager.getCameraCharacteristics(cameraId);
                int facing = characteristics.get(CameraCharacteristics.LENS_FACING);
                if (facing == mCameraFacing) {
                    Log.i(TAG, "camera: " + cameraId + " is selected");
                    Log.i(TAG, "lens facing: " + (facing == 0 ? "front" : (facing == 1 ? "back" : "external")));
                    selected = cameraId;
                    break;
                }
            }
            // Select the first one if no proper cameras
            if (selected == null) {
                selected = cameraIds[0];
                characteristics = manager.getCameraCharacteristics(selected);
                mCameraFacing = characteristics.get(CameraCharacteristics.LENS_FACING);
                Log.i(TAG, "camera: " + selected + " is selected");
                Log.i(TAG, "lens facing: " + (mCameraFacing == 0 ? "front" : (mCameraFacing == 1 ? "back" : "external")));
            }

            // Clockwise angle through which the output image needs to be rotated to be upright
            // on the device screen in its native orientation.
            mSensorOrientation = characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION);
            Log.i(TAG, "sensor orientation: " + mSensorOrientation);

            //  Stream configuration map
            StreamConfigurationMap map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
            int[] formats = map.getOutputFormats();
            // For debugging supported output formats & sizes
            for (int format : formats) {
                StringBuilder outputFormat = new StringBuilder();
                outputFormat.append("output format: " + format + " size: ");
                Size[] sizes = map.getOutputSizes(format);
                for (Size s : sizes) {
                    outputFormat.append(" " + s.getWidth() + "x" + s.getHeight());
                }
                Log.d(TAG, outputFormat.toString());
            }

            // Select output format
            if (!map.isOutputSupportedFor(mOutputFormat)) {
                if (map.isOutputSupportedFor(PixelFormat.RGBA_8888)) {
                    mOutputFormat = PixelFormat.RGBA_8888;
                } else if (map.isOutputSupportedFor(ImageFormat.JPEG)) {
                    mOutputFormat = ImageFormat.JPEG;
                } else if (map.isOutputSupportedFor(ImageFormat.YUV_420_888)) {
                    mOutputFormat = ImageFormat.YUV_420_888;
                } else {
                    mOutputFormat = formats[0];
                }
            }

            // Select output size
            Size[] sizes = map.getOutputSizes(mOutputFormat);
            Size size = sizes[0];
            for (Size s : sizes) {
                int area = s.getWidth() * s.getHeight();
                if (area <= mOutputWidth * mOutputHeight) {
                    if (size.getWidth() * size.getHeight() > mOutputWidth * mOutputHeight) {
                        size = s;
                    }
                    if (size.getWidth() * size.getHeight() < area) {
                        size = s;
                    }
                }
            }
            mOutputWidth = size.getWidth();
            mOutputHeight = size.getHeight();
            Log.i(TAG, "output size: " + mOutputWidth + "x" + mOutputHeight + "is selected");

            // NOTES:
            // In most case, you can pass a RGBA_8888 SurfaceView as the output for camera review.
            // So, RGBA_8888 is impliedly supported by most mobile phone cameras (excepts imx8)
            if (cameraIds.length > 1) {
                mOutputFormat = PixelFormat.RGBA_8888;
            }
            Log.i(TAG, "output format: " + mOutputFormat + " is selected");
        } catch (CameraAccessException e) {
            selected = null;
            e.printStackTrace();
        }
        return selected;
    }

    private void startBackgroundThread() {
        if (mBackgroundHandler == null) {
            mBackgroundThread = new HandlerThread("CameraBackground");
            mBackgroundThread.start();
            mBackgroundHandler = new Handler(mBackgroundThread.getLooper());
        }
    }

    private void stopBackgroundThread() {
        if (mBackgroundThread != null) {
            mBackgroundThread.quitSafely();
            try {
                mBackgroundThread.join();
                mBackgroundThread = null;
                mBackgroundHandler = null;
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }

}
