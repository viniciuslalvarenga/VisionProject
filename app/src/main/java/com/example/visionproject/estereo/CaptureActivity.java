package com.example.visionproject.estereo;

import android.annotation.SuppressLint;
import android.content.Intent;
import android.hardware.camera2.CaptureRequest;
import android.os.Bundle;
import android.text.InputType;
import android.util.Log;
import android.util.Size;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.camera2.interop.Camera2CameraControl;
import androidx.camera.camera2.interop.CaptureRequestOptions;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureException;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.ViewModelProvider;

import com.example.visionproject.R;
import com.example.visionproject.estereo.repository.StereoCsvLogger;
import com.example.visionproject.estereo.viewmodel.CaptureViewModel;
import com.google.common.util.concurrent.ListenableFuture;

import java.io.File;

public class CaptureActivity extends AppCompatActivity {
    private static final String TAG = "CaptureActivity";
    private PreviewView previewView;
    private TextView tvStatus, tvBaseline;
    private Button btnCaptureL, btnCaptureR, btnClear, btnProcess;

    private CaptureViewModel viewModel;
    private ImageCapture imageCapture;
    private ProcessCameraProvider cameraProvider;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.est_activity_capture);

        viewModel = new ViewModelProvider(this).get(CaptureViewModel.class);

        previewView = findViewById(R.id.preview_view);
        tvStatus = findViewById(R.id.tv_capture_status);
        tvBaseline = findViewById(R.id.tv_baseline_display);
        btnCaptureL = findViewById(R.id.btn_capture_l);
        btnCaptureR = findViewById(R.id.btn_capture_r);
        btnClear = findViewById(R.id.btn_clear);
        btnProcess = findViewById(R.id.btn_process_now);

        setupCamera();
        observeViewModel();

        btnCaptureL.setOnClickListener(v -> captureImage("L"));
        btnCaptureR.setOnClickListener(v -> captureImage("R"));
        btnClear.setOnClickListener(v -> viewModel.clear());
        btnProcess.setOnClickListener(v -> {
            viewModel.finalizePair();
            startActivity(new Intent(this, ProcessingActivity.class));
            finish();
        });

        View btnEditBaseline = findViewById(R.id.btn_edit_baseline);
        if (btnEditBaseline != null) {
            btnEditBaseline.setOnClickListener(v -> showBaselineDialog());
        }
    }

    private void showBaselineDialog() {
        final EditText input = new EditText(this);
        input.setInputType(InputType.TYPE_CLASS_NUMBER | InputType.TYPE_NUMBER_FLAG_DECIMAL);
        Float cur = viewModel.getBaselineMm().getValue();
        input.setText(String.valueOf(cur == null ? 60.0f : cur));
        input.setHint("Ex: 450mm ou 45cm (20mm - 1000mm)");
        new AlertDialog.Builder(this)
                .setTitle("Baseline fisica do suporte")
                .setMessage("Meca com regua a distancia entre POS 1 e POS 2 do trilho. Use mm, cm ou m.")
                .setView(input)
                .setPositiveButton("Salvar", (d, w) -> {
                    try {
                        float baseline = parseBaselineValue(input.getText().toString());
                        if (baseline <= 0) throw new NumberFormatException();
                        viewModel.setBaselineMm(baseline);
                    } catch (NumberFormatException ex) {
                        Toast.makeText(this, "Valor invalido. Use por exemplo 450mm ou 45cm.", Toast.LENGTH_SHORT).show();
                    }
                })
                .setNegativeButton("Cancelar", null)
                .show();
    }

    private float parseBaselineValue(String rawValue) {
        if (rawValue == null) return -1f;
        String value = rawValue.trim().toLowerCase().replace(',', '.');
        if (value.isEmpty()) return -1f;

        float parsed;
        if (value.endsWith("mm")) {
            parsed = Float.parseFloat(value.replace("mm", "").trim());
        } else if (value.endsWith("cm")) {
            parsed = Float.parseFloat(value.replace("cm", "").trim()) * 10f;
        } else if (value.endsWith("m")) {
            parsed = Float.parseFloat(value.replace("m", "").trim()) * 1000f;
        } else {
            parsed = Float.parseFloat(value);
        }
        return parsed;
    }

    private void setupCamera() {
        ListenableFuture<ProcessCameraProvider> future = ProcessCameraProvider.getInstance(this);
        future.addListener(() -> {
            try {
                cameraProvider = future.get();
                Preview preview = new Preview.Builder()
                        .setTargetResolution(new Size(1280, 720))
                        .build();
                preview.setSurfaceProvider(previewView.getSurfaceProvider());

                imageCapture = new ImageCapture.Builder()
                        .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
                        .setTargetResolution(new Size(1280, 720))
                        .build();

                CameraSelector selector = CameraSelector.DEFAULT_BACK_CAMERA;
                Camera camera = cameraProvider.bindToLifecycle(this, selector, preview, imageCapture);
                lockCameraSettings(camera);
            } catch (Exception e) {
                Log.e(TAG, "Erro setup CameraX", e);
            }
        }, ContextCompat.getMainExecutor(this));
    }

    @SuppressLint("UnsafeOptInUsageError")
    private void lockCameraSettings(Camera camera) {
        Camera2CameraControl c2 = Camera2CameraControl.from(camera.getCameraControl());
        CaptureRequestOptions opt = new CaptureRequestOptions.Builder()
                .setCaptureRequestOption(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE)
                .setCaptureRequestOption(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON)
                .setCaptureRequestOption(CaptureRequest.CONTROL_AWB_MODE, CaptureRequest.CONTROL_AWB_MODE_AUTO)
                .build();
        c2.setCaptureRequestOptions(opt);
    }

    private void observeViewModel() {
        viewModel.getState().observe(this, state -> {
            switch (state) {
                case AWAIT_LEFT:
                    tvStatus.setText(R.string.est_msg_pos_left);
                    btnCaptureL.setEnabled(true);
                    btnCaptureR.setEnabled(false);
                    btnProcess.setEnabled(false);
                    break;
                case AWAIT_RIGHT:
                    tvStatus.setText(R.string.est_msg_pos_right);
                    btnCaptureL.setEnabled(false);
                    btnCaptureR.setEnabled(true);
                    btnProcess.setEnabled(false);
                    break;
                case PAIR_READY:
                    tvStatus.setText(R.string.est_msg_pair_ready);
                    btnCaptureL.setEnabled(false);
                    btnCaptureR.setEnabled(false);
                    btnProcess.setEnabled(true);
                    break;
            }
        });

        viewModel.getBaselineMm().observe(this, b ->
                tvBaseline.setText(getString(R.string.est_btn_baseline, b.intValue())));
    }

    private void captureImage(String side) {
        File outDir = new File(getExternalFilesDir(null), "estereo_pairs");
        if (!outDir.exists()) outDir.mkdirs();
        File outFile = new File(outDir, "I_" + side + "_" + System.currentTimeMillis() + ".jpg");

        ImageCapture.OutputFileOptions opts =
                new ImageCapture.OutputFileOptions.Builder(outFile).build();

        imageCapture.takePicture(opts, ContextCompat.getMainExecutor(this),
                new ImageCapture.OnImageSavedCallback() {
                    @Override
                    public void onImageSaved(@NonNull ImageCapture.OutputFileResults r) {
                        viewModel.onImageSaved(side, outFile);
                        StereoCsvLogger.getInstance().log(
                                "PAIR_CAPTURED_" + side, outFile.getAbsolutePath());
                        Toast.makeText(CaptureActivity.this,
                                "Capturado: " + side, Toast.LENGTH_SHORT).show();
                    }
                    @Override
                    public void onError(@NonNull ImageCaptureException e) {
                        Toast.makeText(CaptureActivity.this,
                                "Erro: " + e.getMessage(), Toast.LENGTH_SHORT).show();
                    }
                });
    }
}
