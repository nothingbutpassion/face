package com.hsae.dms;

import android.Manifest;
import android.app.ActionBar;
import android.app.Activity;
import android.content.Context;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.graphics.PixelFormat;
import android.media.Image;
import android.os.Bundle;
import android.os.Environment;
import android.support.v4.app.ActivityCompat;
import android.util.Log;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.widget.Toast;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

public class MainActivity extends Activity {
    private final String TAG = "dms";
    private ImageProcessor mImageProcessor = new ImageProcessor();
    private VideoCapture mVideoCapture = new VideoCapture(MainActivity.this);
    private Surface mPreviewSurface;
    private VideoCapture.CaptureListener mCaptureListener = new VideoCapture.CaptureListener() {
        @Override
        public void onCaptured(Image image) {
            long t = System.currentTimeMillis();
            NativeBuffer nativeBuffer = NativeBuffer.fromImage(image);
            Log.i(TAG, "create native buffer from image: " + (System.currentTimeMillis() - t) + "ms");
            processFrame(nativeBuffer);
        }
    };

    private SurfaceHolder.Callback mSurfaceCallback = new SurfaceHolder.Callback() {
        @Override
        public void surfaceCreated(SurfaceHolder holder) {
            Log.i(TAG, "surface created.");
            mPreviewSurface = holder.getSurface();
            mVideoCapture.open();
            mVideoCapture.setCaptureListener(mCaptureListener);
        }
        @Override
        public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
            Log.i(TAG, "surface changed: width=" + width + " height=" + height + " format=" + format);
        }
        @Override
        public void surfaceDestroyed(SurfaceHolder holder) {
            mVideoCapture.close();
            mPreviewSurface = null;
            Log.i(TAG, "surface destroyed.");
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // Check and request permissions if not granted
        requestPermissions();

        // Copy assets files to external storage directory
        copyFiles(this);

        // Enables regular immersive mode.
        // For "lean back" mode, remove SYSTEM_UI_FLAG_IMMERSIVE.
        // Or for "sticky immersive," replace it with SYSTEM_UI_FLAG_IMMERSIVE_STICKY
        // View decorView = getWindow().getDecorView();
        // decorView.setSystemUiVisibility(
        //                View.SYSTEM_UI_FLAG_IMMERSIVE
        //                // Set the content to appear under the system bars so that the
        //                // content doesn't resize when the system bars hide and show.
        //                | View.SYSTEM_UI_FLAG_LAYOUT_STABLE
        //                | View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
        //                | View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
        //                // Hide the nav bar and status bar
        //                | View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
        //                | View.SYSTEM_UI_FLAG_FULLSCREEN);
        // Set screen orientation as portrait
        // setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);

        // Remember that you should never show the action bar
        getWindow().getDecorView().setSystemUiVisibility(View.SYSTEM_UI_FLAG_FULLSCREEN);
        ActionBar actionBar = getActionBar();
        if (actionBar != null) {
            actionBar.hide();
        }
        setContentView(R.layout.activity_main);
        SurfaceView surfaceView = findViewById(R.id.preview);
        surfaceView.getHolder().setFormat(PixelFormat.RGBA_8888);
        surfaceView.getHolder().addCallback(mSurfaceCallback);
        mImageProcessor.open();

    }

    @Override
    protected void onDestroy() {
        mImageProcessor.close();
        super.onDestroy();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        for (int i=0; i < grantResults.length; ++i) {
            if (grantResults[i] != PackageManager.PERMISSION_GRANTED ) {
                Toast.makeText(this, permissions[i] + " is not granted!", Toast.LENGTH_LONG).show();
                finish();
                return;
            }
        }
        // Reload the activity with permission granted
        Toast.makeText(this, "Permission is granted.", Toast.LENGTH_LONG).show();
        finish();
        startActivity(getIntent());
    }

    static {
        System.loadLibrary("dms");
    }

    private void processFrame(NativeBuffer nativeBuffer) {
        long t = System.currentTimeMillis();
        nativeBuffer = nativeBuffer.rotate(mVideoCapture.getCaptureRotation());
        Log.d(TAG, "rotate: " + (System.currentTimeMillis() - t) + "ms");

        t = System.currentTimeMillis();
        nativeBuffer = nativeBuffer.flip(mVideoCapture.getCaptureFlipping());
        Log.d(TAG, "flip: " + (System.currentTimeMillis() - t) + "ms");

        t = System.currentTimeMillis();
        mImageProcessor.process(nativeBuffer);
        Log.d(TAG, "process: " + (System.currentTimeMillis() - t) + "ms");

        t = System.currentTimeMillis();
        if (mPreviewSurface != null && mPreviewSurface.isValid()) {
            nativeBuffer.draw(mPreviewSurface);
        }

        Log.d(TAG, "draw: " + (System.currentTimeMillis() - t) + "ms");
    }

    private  boolean requestPermissions() {
        boolean granted = true;
        String[] permissisons = new String[]{
                Manifest.permission.CAMERA,
                Manifest.permission.READ_EXTERNAL_STORAGE,
                Manifest.permission.WRITE_EXTERNAL_STORAGE
        };
        // Check permissions
        for (String permission: permissisons) {
            if (ActivityCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
                granted = false;
            }
        }
        // Request permissions
        if (!granted) {
            ActivityCompat.requestPermissions(this, permissisons, 0);
            return false;
        }
        return true;
    }

    public static boolean copyFile(Context context, String assetFile) {
        AssetManager assetManager = context.getAssets();
        String saveDir = Environment.getExternalStorageDirectory() + File.separator + "dms";
        String saveFile = saveDir +  File.separator + assetFile;
        if (new File(saveFile).isFile()) {
            return true;
        }
        File diretory = new File(saveDir);
        if (!diretory.exists()) {
            diretory.mkdir();
        }
        try {
            InputStream inputStream = assetManager.open(assetFile);
            FileOutputStream outputStream = new FileOutputStream(saveFile);
            byte[] bytes = new byte[inputStream.available()];
            inputStream.read(bytes);
            inputStream.close();
            outputStream.write(bytes);
            outputStream.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return new File(saveFile).isFile();
    }

    public static boolean copyFiles(Context context) {
        String[] assetFiles = new String[] {
            "kazemi_face_landmark.dat",
            "smoking_classifier.tflite",
            "calling_classifier.tflite",
        };
        boolean result = true;
        for(String file: assetFiles) {
            if (!copyFile(context, file)) {
                result = false;
            }
        }
        return result;
    }
}
