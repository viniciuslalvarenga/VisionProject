package com.example.visionproject;

import android.Manifest;
import android.content.ContentValues;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.Looper;
import android.provider.MediaStore;
import android.util.Log;
import android.view.SurfaceView;
import android.view.WindowManager;
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
import java.io.PrintWriter;
import java.util.Locale;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "MainActivity";
    private static final int CAMERA_PERMISSION_REQUEST_CODE = 100;

    private CameraBridgeViewBase mOpenCvCameraView;
    private PccModule mPccModule;
    private TextView mTvDebugPcc, mTvDebugStatus, mTvDebugDiscard, mTvDebugIta, mTvTimer;
    private boolean mSaveNextFrame = false;
    private volatile int mCannyThreshold = 50;
    private volatile int mViewMode = 0; // 0: Original, 1: Canny

    private volatile boolean mIsExpRunning = false;
    private long mExpStartTime = 0;
    private volatile long mLastUiUpdate = 0; // Para limitar atualizações da UI
    private final StringBuilder mLogBuffer = new StringBuilder();
    private final Handler mTimerHandler = new Handler(Looper.getMainLooper());
    private Runnable mTimerRunnable;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        if (!OpenCVLoader.initLocal()) {
            Log.e(TAG, "OpenCV library not found!");
        }

        setContentView(R.layout.activity_main);

        mOpenCvCameraView = findViewById(R.id.camera_view);
        if (mOpenCvCameraView != null) {
            mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
            mOpenCvCameraView.setCvCameraViewListener(this);
        }

        mPccModule = new PccModule();

        mTvDebugPcc = findViewById(R.id.tv_debug_pcc);
        mTvDebugStatus = findViewById(R.id.tv_debug_status);
        mTvDebugDiscard = findViewById(R.id.tv_debug_discard);
        mTvDebugIta = findViewById(R.id.tv_debug_ita);
        mTvTimer = findViewById(R.id.tv_timer);

        SeekBar sbThreshold = findViewById(R.id.sb_threshold);
        TextView tvThreshold = findViewById(R.id.tv_threshold);
        if (sbThreshold != null && tvThreshold != null) {
            sbThreshold.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
                @Override
                public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                    mCannyThreshold = progress;
                    tvThreshold.setText(getString(R.string.canny_threshold_label, mCannyThreshold));
                }
                @Override public void onStartTrackingTouch(SeekBar seekBar) {}
                @Override public void onStopTrackingTouch(SeekBar seekBar) {}
            });
        }

        SeekBar sbTheta = findViewById(R.id.sb_theta);
        TextView tvThetaLabel = findViewById(R.id.tv_theta_label);
        if (sbTheta != null && tvThetaLabel != null) {
            sbTheta.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
                @Override
                public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                    double theta = progress / 100.0;
                    mPccModule.setThresholdTheta(theta);
                    tvThetaLabel.setText(getString(R.string.pcc_threshold_label, theta));
                }
                @Override public void onStartTrackingTouch(SeekBar seekBar) {}
                @Override public void onStopTrackingTouch(SeekBar seekBar) {}
            });
        }

        findViewById(R.id.btn_reset).setOnClickListener(v -> mPccModule.resetStats());
        
        findViewById(R.id.rb_view_orig).setOnClickListener(v -> mViewMode = 0);
        findViewById(R.id.rb_view_canny).setOnClickListener(v -> mViewMode = 1);

        ToggleButton btnRecord = findViewById(R.id.btn_record);
        btnRecord.setOnCheckedChangeListener((v, isChecked) -> {
            if (isChecked) startExperiment();
            else stopExperiment();
        });

        findViewById(R.id.btn_capture).setOnClickListener(v -> mSaveNextFrame = true);

        mTimerRunnable = new Runnable() {
            @Override
            public void run() {
                if (!mIsExpRunning) return;
                long millis = System.currentTimeMillis() - mExpStartTime;
                int seconds = (int) (millis / 1000);
                int minutes = seconds / 60;
                int sec = seconds % 60;

                mTvTimer.setText(String.format(Locale.US, "%02d:%02d", minutes, sec));
                if (seconds >= 120) {
                    btnRecord.setChecked(false);
                    return;
                }
                mTimerHandler.postDelayed(this, 500);
            }
        };

        checkCameraPermission();
    }

    private void startExperiment() {
        mPccModule.resetStats();
        synchronized (mLogBuffer) {
            mLogBuffer.setLength(0);
            mLogBuffer.append("Seconds,PCC,CRE,Status,DiscardRate,ITA_After\n");
        }
        mIsExpRunning = true;
        mExpStartTime = System.currentTimeMillis();
        mTimerHandler.post(mTimerRunnable);
        Toast.makeText(this, "EXPERIMENTO INICIADO: Gravando dados...", Toast.LENGTH_SHORT).show();
    }

    private void stopExperiment() {
        mIsExpRunning = false;
        mTimerHandler.removeCallbacks(mTimerRunnable);
        saveLogToFile();
        mTvTimer.setText(R.string.timer_default);
    }

    private void logData() {
        long elapsedMillis = System.currentTimeMillis() - mExpStartTime;
        double seconds = elapsedMillis / 1000.0;
        
        String entry = String.format(Locale.US, "%.2f,%.4f,%.4f,%s,%.1f,%d\n",
                seconds,
                mPccModule.getCurrentPcc(), mPccModule.getCurrentCre(),
                mPccModule.getStatus(), mPccModule.getDiscardRate(),
                mPccModule.getItaPointsAfter());

        synchronized (mLogBuffer) {
            mLogBuffer.append(entry);
        }
    }

    private void saveLogToFile() {
        String data;
        synchronized (mLogBuffer) {
            if (mLogBuffer.length() == 0) return;
            data = mLogBuffer.toString();
            mLogBuffer.setLength(0);
        }
        try {
            long timestamp = System.currentTimeMillis();
            String fileName = "Exp1_Theta_" + String.format(Locale.US, "%.2f", mPccModule.getThresholdTheta()) + "_" + timestamp + ".csv";
            ContentValues values = new ContentValues();
            values.put(MediaStore.MediaColumns.DISPLAY_NAME, fileName);
            values.put(MediaStore.MediaColumns.MIME_TYPE, "text/csv");
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                values.put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_DOCUMENTS + "/VisionProject");
            }

            Uri uri = getContentResolver().insert(MediaStore.Files.getContentUri("external"), values);
            if (uri != null) {
                try (OutputStream os = getContentResolver().openOutputStream(uri)) {
                    if (os != null) {
                        try (PrintWriter writer = new PrintWriter(os)) {
                            writer.print(data);
                            writer.flush();
                        }
                    }
                }
                Toast.makeText(this, "EXPERIMENTO CONCLUÍDO\nArquivo salvo: " + fileName, Toast.LENGTH_LONG).show();
            }
        } catch (Exception e) {
            Log.e(TAG, "Erro ao salvar log", e);
            Toast.makeText(this, "Erro ao salvar arquivo!", Toast.LENGTH_SHORT).show();
        }
    }

    private void checkCameraPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION_REQUEST_CODE);
        } else {
            mOpenCvCameraView.setCameraPermissionGranted();
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == CAMERA_PERMISSION_REQUEST_CODE && grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            mOpenCvCameraView.setCameraPermissionGranted();
            mOpenCvCameraView.enableView();
        }
    }

    @Override
    public void onPause() {
        if (mOpenCvCameraView != null) mOpenCvCameraView.disableView();
        super.onPause();
    }

    @Override
    public void onResume() {
        super.onResume();
        if (mOpenCvCameraView != null) {
            if (OpenCVLoader.initLocal()) {
                mOpenCvCameraView.enableView();
            }
        }
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null) mOpenCvCameraView.disableView();
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
    }

    @Override
    public void onCameraViewStopped() {
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        Mat rgba = inputFrame.rgba();

        if (mIsExpRunning) {
            boolean shouldFullProcess = mPccModule.processFrame(rgba);
            
            // Só atualiza a interface a cada 200ms para não travar o app
            long currentTime = System.currentTimeMillis();
            if (currentTime - mLastUiUpdate > 200) {
                updateDebugUI();
                mLastUiUpdate = currentTime;
            }

            if (shouldFullProcess) {
                if (mViewMode == 1) {
                    // Modo Canny
                    Mat gray = new Mat(), blur = new Mat(), edges = new Mat();
                    Imgproc.cvtColor(rgba, gray, Imgproc.COLOR_RGBA2GRAY);
                    Imgproc.GaussianBlur(gray, blur, new Size(5, 5), 0);
                    Imgproc.Canny(blur, edges, mCannyThreshold, mCannyThreshold * 2);
                    Imgproc.cvtColor(edges, rgba, Imgproc.COLOR_GRAY2RGBA);
                    
                    if (mSaveNextFrame) {
                        mSaveNextFrame = false;
                        savePipeline(rgba.clone(), gray.clone(), blur.clone(), edges.clone());
                    }
                    gray.release(); blur.release(); edges.release();
                } else if (mSaveNextFrame) {
                    mSaveNextFrame = false;
                    saveFrame(rgba.clone());
                }
            } else {
                // Caso de descarte (DPM ativado)
                if (mSaveNextFrame) {
                    mSaveNextFrame = false;
                    saveFrame(rgba.clone());
                }
            }
            logData();
        } else if (mSaveNextFrame) {
            mSaveNextFrame = false;
            saveFrame(rgba.clone());
        }

        return rgba;
    }

    private void updateDebugUI() {
        runOnUiThread(() -> {
            String fullStatus = mPccModule.getFullStatus();
            double pcc = mPccModule.getCurrentPcc();
            double cre = mPccModule.getCurrentCre();
            double discardRate = mPccModule.getDiscardRate();
            int ita = mPccModule.getItaPointsAfter();

            mTvDebugPcc.setText(getString(R.string.pcc_cre_label, pcc, cre));
            mTvDebugStatus.setText(getString(R.string.status_label, fullStatus));
            
            String status = mPccModule.getStatus();
            if ("IDLE".equals(status)) mTvDebugStatus.setTextColor(Color.WHITE);
            else if ("DISCARD".equals(status)) mTvDebugStatus.setTextColor(Color.YELLOW);
            else mTvDebugStatus.setTextColor(Color.GREEN);

            mTvDebugDiscard.setText(String.format(Locale.US, "Descarte: %.1f%%", discardRate));
            mTvDebugIta.setText(String.format(Locale.US, "ITA: %d pts", ita));
        });
    }

    private void savePipeline(Mat orig, Mat gray, Mat blur, Mat edges) {
        long ts = System.currentTimeMillis();
        saveFrame(orig, "p_orig_" + ts + ".jpg");
        saveFrame(gray, "p_gray_" + ts + ".jpg");
        saveFrame(blur, "p_blur_" + ts + ".jpg");
        saveFrame(edges, "p_edge_" + ts + ".jpg");
        runOnUiThread(() -> Toast.makeText(this, "Pipeline Salvo", Toast.LENGTH_SHORT).show());
    }

    private void saveFrame(Mat frame) { saveFrame(frame, "cap_" + System.currentTimeMillis() + ".jpg"); }

    private void saveFrame(Mat frame, String filename) {
        Bitmap bmp = Bitmap.createBitmap(frame.cols(), frame.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(frame, bmp);
        ContentValues v = new ContentValues();
        v.put(MediaStore.Images.Media.DISPLAY_NAME, filename);
        v.put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg");
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            v.put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/VisionProject");
        }
        Uri uri = getContentResolver().insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, v);
        if (uri != null) {
            try (OutputStream out = getContentResolver().openOutputStream(uri)) {
                if (out != null) {
                    bmp.compress(Bitmap.CompressFormat.JPEG, 90, out);
                }
            } catch (Exception e) { Log.e(TAG, "Err save", e); }
        }
        frame.release();
    }
}
