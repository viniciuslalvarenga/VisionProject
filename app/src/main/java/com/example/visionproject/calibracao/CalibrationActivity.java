package com.example.visionproject.calibracao;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.ToggleButton;

import androidx.activity.OnBackPressedCallback;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.ViewModelProvider;

import com.example.visionproject.R;
import com.example.visionproject.calibracao.model.CalibrationState;
import com.example.visionproject.calibracao.ui.ChessboardOverlayView;
import com.example.visionproject.calibracao.ui.CoverageHeatmapView;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;

public class CalibrationActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final int CAMERA_PERMISSION_REQUEST_CODE = 200;
    
    private CalibrationViewModel viewModel;
    private CameraBridgeViewBase cameraView;
    private ChessboardOverlayView overlayView;
    private CoverageHeatmapView heatmapView;
    private TextView tvStatus, tvCount, tvBlur;
    private Button btnCalibrate, btnSave, btnUndistort;
    private ProgressBar progressBar;
    private Mat lastFrame = new Mat();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.cal_activity_calibracao);

        initViews();
        setupViewModel();
        setupListeners();
        checkCameraPermission();

        getOnBackPressedDispatcher().addCallback(this, new OnBackPressedCallback(true) {
            @Override
            public void handleOnBackPressed() {
                finish();
            }
        });
    }

    private void initViews() {
        cameraView = findViewById(R.id.cal_camera_view);
        cameraView.setVisibility(SurfaceView.VISIBLE);
        cameraView.setCvCameraViewListener(this);

        overlayView = findViewById(R.id.cal_overlay_view);
        heatmapView = findViewById(R.id.cal_heatmap_view);
        tvStatus = findViewById(R.id.cal_tv_status);
        tvCount = findViewById(R.id.cal_tv_count);
        tvBlur = findViewById(R.id.cal_tv_blur);
        btnCalibrate = findViewById(R.id.cal_btn_calibrate);
        btnSave = findViewById(R.id.cal_btn_save);
        btnUndistort = findViewById(R.id.cal_btn_undistort);
        progressBar = findViewById(R.id.cal_progress);
    }

    private void setupViewModel() {
        viewModel = new ViewModelProvider(this).get(CalibrationViewModel.class);

        viewModel.getState().observe(this, state -> {
            tvStatus.setText("Status: " + state.name());
            // Agora permite calibrar se estiver pronto, se já terminou ou se deu erro antes
            btnCalibrate.setEnabled(state == CalibrationState.READY_TO_CALIBRATE || 
                                  state == CalibrationState.DONE || 
                                  state == CalibrationState.ERROR);

            btnSave.setEnabled(state == CalibrationState.DONE);
            btnUndistort.setEnabled(state == CalibrationState.DONE);
            progressBar.setVisibility(state == CalibrationState.CALIBRATING ? View.VISIBLE : View.GONE);
            
            if (state == CalibrationState.CAPTURED) {
                Toast.makeText(this, "Frame capturado!", Toast.LENGTH_SHORT).show();
            }
        });

        viewModel.getFramesCollectedCount().observe(this, count -> {
            tvCount.setText(String.format("Frames: %d/15", count));
            heatmapView.updateCoverage(
                com.example.visionproject.calibracao.repository.CalibrationFramesRepository.getInstance().getAll(),
                cameraView.getWidth(),
                cameraView.getHeight()
            );
        });

        viewModel.getLastBlurScore().observe(this, score -> {
            tvBlur.setText(String.format("Blur Score: %.1f", score));
        });

        viewModel.getUserMessage().observe(this, msg -> {
            if (msg != null && !msg.isEmpty()) {
                Toast.makeText(this, msg, Toast.LENGTH_LONG).show();
            }
        });

        viewModel.getLastDetection().observe(this, detection -> {
            overlayView.setDetection(detection, viewModel.getState().getValue());
        });

        viewModel.getResult().observe(this, result -> {
            if (result != null) {
                String summary = String.format(java.util.Locale.US,
                        "RMS: %.3f px\nfx: %.1f fy: %.1f\ncx: %.1f cy: %.1f",
                        result.getRms(), result.getFx(), result.getFy(), result.getCx(), result.getCy());
                tvStatus.setText(summary);
            }
        });

        viewModel.getUndistortComparison().observe(this, pair -> {
            if (pair != null) {
                showComparisonDialog(pair.first, pair.second);
            }
        });
    }

    private void showComparisonDialog(android.graphics.Bitmap original, android.graphics.Bitmap corrected) {
        android.app.Dialog dialog = new android.app.Dialog(this, android.R.style.Theme_Black_NoTitleBar_Fullscreen);
        dialog.setContentView(R.layout.cal_dialog_comparison);
        
        com.example.visionproject.modelocamera.ui.ComparisonImageView comparisonView = dialog.findViewById(R.id.cal_comparison_view);
        comparisonView.setOriginal(original);
        comparisonView.setCorrected(corrected);
        
        dialog.findViewById(R.id.cal_btn_close_comparison).setOnClickListener(v -> dialog.dismiss());
        dialog.show();
    }

    private void setupListeners() {
        findViewById(R.id.cal_btn_capture).setOnClickListener(v -> {
            if (!lastFrame.empty()) {
                viewModel.captureFrame(lastFrame);
            } else {
                Toast.makeText(this, "Câmera ainda não iniciada", Toast.LENGTH_SHORT).show();
            }
        });
        
        ToggleButton btnAuto = findViewById(R.id.cal_btn_auto);
        btnAuto.setOnCheckedChangeListener((v, isChecked) -> {
            if (viewModel.getAutoCaptureEnabled().getValue() != isChecked) {
                viewModel.toggleAutoCapture();
            }
        });

        findViewById(R.id.cal_btn_clear).setOnClickListener(v -> viewModel.clearFrames());

        findViewById(R.id.cal_btn_list).setOnClickListener(v -> {
            new com.example.visionproject.calibracao.ui.FramesListDialog(viewModel).show(getSupportFragmentManager(), "frames_list");
        });

        btnCalibrate.setOnClickListener(v -> {
            viewModel.runCalibration();
        });

        btnSave.setOnClickListener(v -> viewModel.saveResult());

        btnUndistort.setOnClickListener(v -> {
            if (!lastFrame.empty()) {
                viewModel.generateUndistortPreview(lastFrame);
            } else {
                Toast.makeText(this, "Nenhum frame disponível para comparação", Toast.LENGTH_SHORT).show();
            }
        });

        findViewById(R.id.cal_btn_back).setOnClickListener(v -> finish());
    }

    private void checkCameraPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION_REQUEST_CODE);
        } else {
            cameraView.setCameraPermissionGranted();
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == CAMERA_PERMISSION_REQUEST_CODE && grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            cameraView.setCameraPermissionGranted();
        }
    }

    @Override
    public void onPause() {
        super.onPause();
        if (cameraView != null) cameraView.disableView();
    }

    @Override
    public void onResume() {
        super.onResume();
        if (cameraView != null) {
            if (OpenCVLoader.initLocal()) {
                cameraView.enableView();
            }
        }
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        if (cameraView != null) cameraView.disableView();
    }

    @Override
    public void onCameraViewStarted(int width, int height) {}

    @Override
    public void onCameraViewStopped() {}

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        Mat rgba = inputFrame.rgba();
        rgba.copyTo(lastFrame);
        viewModel.onPreviewFrame(rgba);
        return rgba;
    }
}
