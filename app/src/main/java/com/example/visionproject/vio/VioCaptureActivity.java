package com.example.visionproject.vio;

import android.content.Intent;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.util.Size;
import android.widget.Button;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.ViewModelProvider;

import com.example.visionproject.R;
import com.example.visionproject.vio.repository.VioCsvLogger;
import com.example.visionproject.vio.sensor.SensorLogger;
import com.example.visionproject.vio.ui.ReadinessBarView;
import com.example.visionproject.vio.viewmodel.VioCaptureViewModel;
import com.google.common.util.concurrent.ListenableFuture;

import org.opencv.core.Mat;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class VioCaptureActivity extends AppCompatActivity {
    private static final String TAG = "VioCaptureActivity";

    private VioCaptureViewModel viewModel;
    private SensorLogger sensorLogger;
    private SensorManager sensorManager;
    private ExecutorService cameraExecutor;

    private TextView tvStatus, tvOffset, tvState;
    private ReadinessBarView readinessBar;
    private Button btnStart, btnPause, btnCapture, btnProcess, btnImuTest;

    private volatile float[] lastRotation = {0, 0, 0, 1};
    private volatile long lastSensorTns = 0;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.vio_activity_capture);

        viewModel = new ViewModelProvider(this).get(VioCaptureViewModel.class);
        sensorManager = (SensorManager) getSystemService(SENSOR_SERVICE);
        cameraExecutor = Executors.newSingleThreadExecutor();

        tvStatus    = findViewById(R.id.tv_vio_state);
        tvOffset    = findViewById(R.id.tv_vio_offset);
        tvState     = tvStatus;
        readinessBar = findViewById(R.id.vio_readiness_bar);
        btnStart    = findViewById(R.id.btn_vio_start);
        btnPause    = findViewById(R.id.btn_vio_pause);
        btnCapture  = findViewById(R.id.btn_vio_capture_pair);
        btnProcess  = findViewById(R.id.btn_vio_process);
        btnImuTest  = findViewById(R.id.btn_vio_imu_test);

        setupSensorLogger();
        setupCamera();
        observeViewModel();
        setupButtons();
    }

    private void setupSensorLogger() {
        sensorLogger = new SensorLogger();
        sensorLogger.setListener(new SensorLogger.Listener() {
            @Override public void onAccel(SensorLogger.Sample s) {
                lastSensorTns = s.tNs;
                viewModel.onAccelSample(s.tNs, s.v);
            }
            @Override public void onGyro(SensorLogger.Sample s) {}
            @Override public void onRotation(SensorLogger.Sample s) {
                lastSensorTns = s.tNs;
                lastRotation = s.v.clone();
                viewModel.onRotationSample(s.tNs, s.v);
            }
        });
    }

    private void setupCamera() {
        ListenableFuture<ProcessCameraProvider> future = ProcessCameraProvider.getInstance(this);
        future.addListener(() -> {
            try {
                ProcessCameraProvider provider = future.get();
                PreviewView previewView = findViewById(R.id.vio_preview_view);

                Preview preview = new Preview.Builder()
                        .setTargetResolution(new Size(1280, 720))
                        .build();
                preview.setSurfaceProvider(previewView.getSurfaceProvider());

                ImageAnalysis analysis = new ImageAnalysis.Builder()
                        .setTargetResolution(new Size(1280, 720))
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build();

                analysis.setAnalyzer(cameraExecutor, image -> {
                    long tFrameNs = image.getImageInfo().getTimestamp();
                    Mat rgb = StereoUtils.yuvToRgbMat(image);
                    image.close();
                    if (rgb != null && !rgb.empty()) {
                        viewModel.onNewFrame(tFrameNs, rgb);
                    }
                });

                provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis);

                // Diagnose clock offset once sensors have data
                new Handler(Looper.getMainLooper()).postDelayed(() ->
                        viewModel.diagnoseClockOffset(), 2000);

            } catch (Exception e) {
                Log.e(TAG, "Erro ao configurar câmera", e);
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void observeViewModel() {
        viewModel.getState().observe(this, s -> tvStatus.setText("Estado: " + s.name()));
        viewModel.getStatusMessage().observe(this, msg -> {
            if (tvState != null) tvState.setText(msg);
        });
        viewModel.getBaselineRatio().observe(this, bRatio ->
                readinessBar.setReadiness(bRatio, 0));
        viewModel.getClockOffsetMs().observe(this, offset ->
                tvOffset.setText("Offset câm↔IMU: " + offset + " ms"));
        viewModel.getState().observe(this, state -> {
            boolean ready = state == VioState.KEYFRAME_READY;
            btnProcess.setEnabled(ready);
            btnCapture.setEnabled(state == VioState.CAPTURING);
        });
    }

    private void setupButtons() {
        btnStart.setOnClickListener(v -> viewModel.startCapture());
        btnPause.setOnClickListener(v -> viewModel.pauseCapture());
        btnCapture.setOnClickListener(v -> viewModel.resetReference());
        btnProcess.setOnClickListener(v -> {
            startActivity(new Intent(this, VioProcessingActivity.class));
        });
        btnImuTest.setOnClickListener(v ->
                startActivity(new Intent(this, VioImuTestActivity.class)));
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (sensorLogger != null && sensorManager != null) {
            sensorLogger.start(sensorManager);
            VioCsvLogger.getInstance().log("IMU_STARTED", "sensor resumed");
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (sensorLogger != null && sensorManager != null) {
            sensorLogger.stop(sensorManager);
            VioCsvLogger.getInstance().log("IMU_STOPPED", "sensor paused");
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        cameraExecutor.shutdown();
        VioCsvLogger.getInstance().log("SESSION_CLOSED", "capture activity destroyed");
        VioCsvLogger.getInstance().saveSession(this);
    }
}
