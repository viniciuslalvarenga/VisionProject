package com.example.visionproject;

import android.Manifest;
import android.content.ContentValues;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.view.SurfaceView;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.ToggleButton;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "MainActivity";
    private static final int CAMERA_PERMISSION_REQUEST_CODE = 100;

    private CameraBridgeViewBase mOpenCvCameraView;
    private Mat mRgbaFrame;
    private Mat mIntermediateFrame;
    private boolean mSaveNextFrame = false;
    private int mCannyThreshold = 50;
    private boolean mIsProcessing = true;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);

        mOpenCvCameraView = findViewById(R.id.camera_view);
        if (mOpenCvCameraView != null) {
            mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
            mOpenCvCameraView.setCvCameraViewListener(this);
        }

        SeekBar sbThreshold = findViewById(R.id.sb_threshold);
        TextView tvThreshold = findViewById(R.id.tv_threshold);
        if (sbThreshold != null && tvThreshold != null) {
            sbThreshold.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
                @Override
                public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                    mCannyThreshold = progress;
                    tvThreshold.setText(getString(R.string.canny_threshold, mCannyThreshold));
                }
                @Override public void onStartTrackingTouch(SeekBar seekBar) {}
                @Override public void onStopTrackingTouch(SeekBar seekBar) {}
            });
        }

        ToggleButton btnProcess = findViewById(R.id.btn_process);
        if (btnProcess != null) {
            btnProcess.setOnCheckedChangeListener((buttonView, isChecked) -> mIsProcessing = isChecked);
        }

        Button btnCapture = findViewById(R.id.btn_capture);
        if (btnCapture != null) {
            btnCapture.setOnClickListener(v -> mSaveNextFrame = true);
        }

        checkCameraPermission();
    }

    private void checkCameraPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.CAMERA},
                    CAMERA_PERMISSION_REQUEST_CODE);
        } else {
            mOpenCvCameraView.setCameraPermissionGranted();
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == CAMERA_PERMISSION_REQUEST_CODE) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                mOpenCvCameraView.setCameraPermissionGranted();
                mOpenCvCameraView.enableView();
            } else {
                Toast.makeText(this, R.string.camera_permission_required, Toast.LENGTH_LONG).show();
            }
        }
    }

    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initLocal()) {
            Log.e(TAG, "OpenCV library not found!");
            Toast.makeText(this, R.string.opencv_init_failed, Toast.LENGTH_LONG).show();
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mOpenCvCameraView.enableView();
        }
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        mRgbaFrame = new Mat();
        mIntermediateFrame = new Mat();
    }

    @Override
    public void onCameraViewStopped() {
        if (mRgbaFrame != null) mRgbaFrame.release();
        if (mIntermediateFrame != null) mIntermediateFrame.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mRgbaFrame = inputFrame.rgba();

        if (mIsProcessing) {
            Mat gray = new Mat();
            Mat blur = new Mat();
            Mat edges = new Mat();

            // 1. Cinza
            Imgproc.cvtColor(mRgbaFrame, gray, Imgproc.COLOR_RGBA2GRAY);
            // 2. Gaussiano
            Imgproc.GaussianBlur(gray, blur, new Size(5, 5), 0);
            // 3. Canny
            Imgproc.Canny(blur, edges, mCannyThreshold, mCannyThreshold * 2);

            if (mSaveNextFrame) {
                mSaveNextFrame = false;
                List<Mat> framesToSave = new ArrayList<>();
                framesToSave.add(mRgbaFrame.clone()); // Original
                framesToSave.add(gray.clone());      // Cinza
                framesToSave.add(blur.clone());      // Suavizada
                framesToSave.add(edges.clone());     // Bordas
                savePipeline(framesToSave);
            }

            // Converter bordas de volta para RGBA para exibição
            Imgproc.cvtColor(edges, mRgbaFrame, Imgproc.COLOR_GRAY2RGBA);
            
            gray.release();
            blur.release();
            edges.release();
        } else if (mSaveNextFrame) {
            mSaveNextFrame = false;
            saveFrame(mRgbaFrame.clone());
        }

        return mRgbaFrame;
    }

    private void savePipeline(List<Mat> frames) {
        long timestamp = System.currentTimeMillis();
        String[] labels = {"original", "gray", "blur", "edges"};
        
        for (int i = 0; i < frames.size(); i++) {
            saveFrame(frames.get(i), "pipeline_" + labels[i] + "_" + timestamp + ".jpg");
        }
        runOnUiThread(() -> Toast.makeText(this, R.string.pipeline_saved, Toast.LENGTH_SHORT).show());
    }

    private void saveFrame(Mat frame) {
        saveFrame(frame, "capture_" + System.currentTimeMillis() + ".jpg");
    }

    private void saveFrame(Mat frame, String filename) {
        Bitmap bitmap = Bitmap.createBitmap(frame.cols(), frame.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(frame, bitmap);
        frame.release();

        ContentValues values = new ContentValues();
        values.put(MediaStore.Images.Media.DISPLAY_NAME, filename);
        values.put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg");
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            values.put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/VisionProject");
            values.put(MediaStore.Images.Media.IS_PENDING, 1);
        }

        Uri uri = getContentResolver().insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);

        if (uri != null) {
            try (OutputStream out = getContentResolver().openOutputStream(uri)) {
                if (out != null) {
                    bitmap.compress(Bitmap.CompressFormat.JPEG, 90, out);
                }
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    values.clear();
                    values.put(MediaStore.Images.Media.IS_PENDING, 0);
                    getContentResolver().update(uri, values, null, null);
                }
            } catch (Exception e) {
                Log.e(TAG, "Erro ao salvar frame: " + filename, e);
            }
        }
    }
}