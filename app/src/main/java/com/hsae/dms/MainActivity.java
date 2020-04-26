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
import java.lang.annotation.Native;
import java.util.concurrent.ConcurrentLinkedQueue;

public class MainActivity extends Activity {
    private final String TAG = "face MainActivity";
    private ImageProcessor mImageProcessor = new ImageProcessor();
    private VideoCapture mVideoCapture = new VideoCapture(MainActivity.this);
    private Surface mPreviewSurface;

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
        /*
        View decorView = getWindow().getDecorView();
        decorView.setSystemUiVisibility(
                View.SYSTEM_UI_FLAG_IMMERSIVE
                // Set the content to appear under the system bars so that the
                // content doesn't resize when the system bars hide and show.
                | View.SYSTEM_UI_FLAG_LAYOUT_STABLE
                | View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
                | View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
                // Hide the nav bar and status bar
                | View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                | View.SYSTEM_UI_FLAG_FULLSCREEN);
        */

        // Set screen orientation as portrait
        //setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);

        // Remember that you should never show the action bar
        getWindow().getDecorView().setSystemUiVisibility(View.SYSTEM_UI_FLAG_FULLSCREEN);
        ActionBar actionBar = getActionBar();
        if (actionBar != null) {
            actionBar.hide();
        }

        setContentView(R.layout.activity_main);
        SurfaceView surfaceView = (SurfaceView) findViewById(R.id.preview);
        surfaceView.getHolder().setFormat(PixelFormat.RGBA_8888);
//        surfaceView.setOnClickListener(new View.OnClickListener() {
//            private int cameraDevice = 0;
//            @Override
//            public void onClick(View v) {
//                mVideoCapture.close();
//                cameraDevice = (cameraDevice + 1) % 2;
//                mVideoCapture.setCaptureDevice(cameraDevice);
//                mVideoCapture.open();
//            }
//        });
        surfaceView.getHolder().addCallback(new SurfaceHolder.Callback() {
            @Override
            public void surfaceCreated(SurfaceHolder holder) {
                Log.i(TAG, "surface created.");
                mPreviewSurface = holder.getSurface();
                mVideoCapture.setCaptureListener(new VideoCapture.CaptureListener() {
                    @Override
                    public void onCaptured(Image image) {
                        long t = System.currentTimeMillis();
                        NativeBuffer nativeBuffer = NativeBuffer.fromImage(image);
                        Log.i(TAG, "NativeBuffer.fromImage: " + (System.currentTimeMillis() - t) + "ms");
                        processFrame(nativeBuffer);
                    }
                });
                mVideoCapture.open();
            }
            @Override
            public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
                Log.i(TAG, "surface changed: width=" + width + " height=" + height + " format=" + format);
            }
            @Override
            public void surfaceDestroyed(SurfaceHolder holder) {
                Log.i(TAG, "surface destroyed.");
                mVideoCapture.close();
                mPreviewSurface = null;
            }
        });
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

    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("face");
    }

    private void processFrame(NativeBuffer nativeBuffer) {
        long t1 = System.currentTimeMillis();
        nativeBuffer = nativeBuffer.rotate(mVideoCapture.getCaptureRotation());
        Log.i(TAG, "NativeBuffer.rotate: " + (System.currentTimeMillis() - t1) + "ms");

        long t2 = System.currentTimeMillis();
        nativeBuffer = nativeBuffer.flip(mVideoCapture.getCaptureFlipping());
        Log.i(TAG, "NativeBuffer.flip: " + (System.currentTimeMillis() - t2) + "ms");

        long t3 = System.currentTimeMillis();
        mImageProcessor.process(nativeBuffer);
        Log.i(TAG, "ImageProcessor.process: " + (System.currentTimeMillis() - t3) + "ms");

        long t4 = System.currentTimeMillis();
        if (mPreviewSurface != null && mPreviewSurface.isValid()) {
            nativeBuffer.draw(mPreviewSurface);
        }
        Log.i(TAG, "NativeBuffer.draw: " + (System.currentTimeMillis() - t4) + "ms");
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
        String saveDir = Environment.getExternalStorageDirectory() + File.separator + "Face";
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
                "resnet_face_descriptor.dat"
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
