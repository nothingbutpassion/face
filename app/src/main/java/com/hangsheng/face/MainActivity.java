package com.hangsheng.face;

import android.Manifest;
import android.app.ActionBar;
import android.app.Activity;
import android.content.Context;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.graphics.PixelFormat;
import android.graphics.PointF;
import android.graphics.Rect;
import android.media.Image;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.Message;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.TextView;
import android.widget.Toast;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

public class MainActivity extends Activity {
    private final String TAG = "MainActivity";
    private final int MESSAGE_FACE_DETECTED = 1111;
    private VideoCapture mVideoCapture;
    private FaceDetector mFaceDetector;
    private Surface mPreviewSurface;
    private int mPreviewWidth;
    private int mPreviewHeight;
    private TextView mTextView;
    private String mStatus;
    private Handler mHandler;

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
        int uiOptions = View.SYSTEM_UI_FLAG_FULLSCREEN;
        getWindow().getDecorView().setSystemUiVisibility(uiOptions);
        ActionBar actionBar = getActionBar();
        if (actionBar != null) {
            actionBar.hide();
        }

        setContentView(R.layout.activity_main);
        mTextView = (TextView) findViewById(R.id.status_text);
//        mHandler = new Handler() {
//            @Override
//            public void handleMessage(Message msg) {
//                if (msg.what == MESSAGE_FACE_DETECTED) {
//                    int bufWidth = msg.arg1;
//                    int bufHeight = msg.arg2;
//                    float ration = mPreviewHeight/(float)bufHeight;
//                    if (ration*bufWidth > mPreviewWidth) {
//                        ration = mPreviewWidth/(float)bufWidth;
//                    }
//                    Rect[] rects = (Rect[]) msg.obj;
//                    String text = "Face detected:";
//                    for (Rect r: rects) {
//                        text += "\n[" + (int)ration * r.left + ", " + (int)ration * r.top + ", "
//                                      + (int)ration * r.right + ", " + (int)ration* r.bottom + "]";
//                    }
//                    mStatus = text;
//                } else {
//                    mStatus = "Face detecting ...";
//                }
//                mTextView.setText(mStatus);
//                super.handleMessage(msg);
//            }
//        };

        SurfaceView surfaceView = ((SurfaceView) findViewById(R.id.camera_view));
        surfaceView.setOnClickListener(new View.OnClickListener() {
            private int cameraDevice = 0;
            @Override
            public void onClick(View v) {
                if (mVideoCapture != null) {
                    mVideoCapture.close();
                    cameraDevice = (cameraDevice + 1) % 2;
                    mVideoCapture.setCaptureDevice(cameraDevice);
                    mVideoCapture.open();
                }
            }
        });
        surfaceView.getHolder().addCallback(new SurfaceHolder.Callback() {
            @Override
            public void surfaceCreated(SurfaceHolder holder) {
                mPreviewSurface = holder.getSurface();
            }
            @Override
            public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
                mPreviewWidth = width;
                mPreviewHeight = height;
            }
            @Override
            public void surfaceDestroyed(SurfaceHolder holder) {
                mPreviewSurface = null;
            }
        });
        surfaceView.getHolder().setFormat(PixelFormat.RGBA_8888);
    }

    @Override
    protected void onResume() {
        super.onResume();
        mFaceDetector = new FaceDetector();
        mFaceDetector.open();
        mVideoCapture = new VideoCapture(MainActivity.this);
        mVideoCapture.setCaptureImage(800, 600, PixelFormat.RGBA_8888);
        mVideoCapture.setCaptureListener(new VideoCapture.CaptureListener() {
            @Override
            public void onCaptured(Image image) {
                long t0 = System.currentTimeMillis();
                Log.d(TAG, "captured image: size=" + image.getWidth() + "x" + image.getHeight() + " format=" + image.getFormat());

                // Draw surface
                if (mPreviewSurface != null) {
                    long t1 = System.currentTimeMillis();
                    NativeBuffer nativeBuffer = new NativeBuffer(image);
                    Log.i(TAG, "NativeBuffer.<init>: " + ( System.currentTimeMillis() - t1) + "ms");

                    long t2 = System.currentTimeMillis();
                    nativeBuffer = nativeBuffer.rotate(mVideoCapture.getCaptureRotation());
                    Log.i(TAG, "NativeBuffer.rotate: " + ( System.currentTimeMillis() - t2) + "ms");

                    long t3 = System.currentTimeMillis();
                    nativeBuffer = nativeBuffer.flip(mVideoCapture.getCaptureFlipping());
                    Log.i(TAG, "NativeBuffer.flip: " + ( System.currentTimeMillis() - t3) + "ms");

                    // Face detection;
                    long t4 = System.currentTimeMillis();
                    mFaceDetector.process(nativeBuffer);
                    Log.i(TAG, "FaceDetector.process: " + ( System.currentTimeMillis() - t4) + "ms");

//                    long t3 = System.currentTimeMillis();
//                    Rect[] faces = mFaceDetector.findFaces(nativeBuffer);
//                    Log.i(TAG, "FaceDetector.detect: " + ( System.currentTimeMillis() - t3) + "ms");
//
//                    if (faces.length > 0) {
//                        Message message = Message.obtain();
//                        message.what = MESSAGE_FACE_DETECTED;
//                        message.obj = faces;
//                        message.arg1 = nativeBuffer.getWidth();
//                        message.arg2 = nativeBuffer.getHeight();
//                        mHandler.sendMessage(message);
//                    } else if (mStatus != "") {
//                        mHandler.sendEmptyMessage(0);
//                    }
//
//                    for (int i=0; i < faces.length; ++i) {
//                        long t4 = System.currentTimeMillis();
//                        PointF[] faceMarks = mFaceDetector.getMarks(nativeBuffer, faces[i]);
//                        Log.i(TAG, "FaceDetector.getMarks: " + ( System.currentTimeMillis() - t4) + "ms");
//                    }

                    long t5 = System.currentTimeMillis();
                    nativeBuffer.draw(mPreviewSurface);
                    Log.i(TAG, "NativeBuffer.draw: " + ( System.currentTimeMillis() - t5) + "ms");
                }
                Log.i(TAG, "VideoCapture.onCaptured: " + ( System.currentTimeMillis() - t0) + "ms");
            }
        });
        mVideoCapture.open();
    }

    @Override
    protected void onPause() {
        super.onPause();
        mVideoCapture.close();
        mVideoCapture = null;
        mFaceDetector.close();
        mFaceDetector = null;
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
                "lbfmodel.yaml",
                "res10_300x300_ssd_iter_140000.prototxt",
                "res10_300x300_ssd_iter_140000.caffemodel"
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
