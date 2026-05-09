package com.example.visionproject.modelocamera;

import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.view.View;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.lifecycle.ViewModelProvider;

import com.example.visionproject.R;
import com.example.visionproject.modelocamera.ui.ComparisonImageView;
import com.example.visionproject.modelocamera.ui.EpipolarOverlayView;
import com.example.visionproject.shared.FileExporter;
import com.example.visionproject.shared.ImageCaptureHelper;

import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;

/**
 * Activity principal do Módulo 2 - Modelo de Câmera.
 * Implementa a interface do usuário seguindo o padrão MVVM.
 */
public class ModeloCameraActivity extends AppCompatActivity {

    private ModeloCameraViewModel viewModel;
    private ComparisonImageView comparisonView;
    private EpipolarOverlayView epipolarView;
    private TextView tvParams;
    private TextView tvStatus;

    private final ActivityResultLauncher<Intent> galleryLauncher = registerForActivityResult(
            new ActivityResultContracts.StartActivityForResult(),
            result -> {
                if (result.getResultCode() == RESULT_OK && result.getData() != null) {
                    Uri imageUri = result.getData().getData();
                    if (imageUri != null) {
                        Mat mat = ImageCaptureHelper.uriToMat(this, imageUri);
                        viewModel.onImageCaptured(mat);
                        mat.release();
                    }
                }
            }
    );

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.mc_activity_modelo_camera);

        // Inicialização do OpenCV (reaproveitada)
        if (!OpenCVLoader.initDebug()) {
            Toast.makeText(this, R.string.opencv_init_failed, Toast.LENGTH_LONG).show();
        }

        initViews();
        setupViewModel();
        setupListeners();
    }

    private void initViews() {
        comparisonView = findViewById(R.id.mc_comparison_view);
        epipolarView = findViewById(R.id.mc_epipolar_view);
        tvParams = findViewById(R.id.mc_tv_params);
        tvStatus = findViewById(R.id.mc_tv_status);
    }

    private void setupViewModel() {
        viewModel = new ViewModelProvider(this, new ModeloCameraVMFactory(getApplication())).get(ModeloCameraViewModel.class);

        viewModel.getIntrinsics().observe(this, k -> updateParamsText());
        viewModel.getDistortion().observe(this, d -> updateParamsText());

        viewModel.getOriginalImage().observe(this, bmp -> {
            comparisonView.setOriginal(bmp);
            epipolarView.setBackgroundImage(bmp);
        });

        viewModel.getCorrectedImage().observe(this, bmp -> {
            comparisonView.setCorrected(bmp);
        });

        viewModel.getStatusMessage().observe(this, msg -> {
            tvStatus.setText(msg);
        });
    }

    private void updateParamsText() {
        if (viewModel.getIntrinsics().getValue() != null && viewModel.getDistortion().getValue() != null) {
            String text = viewModel.getIntrinsics().getValue().toString() + "\n" +
                         viewModel.getDistortion().getValue().toString();
            tvParams.setText(text);
        }
    }

    private void setupListeners() {
        comparisonView.setOnPointSelectedListener(p -> {
            viewModel.onPointSelected(p);
        });

        epipolarView.setOnPointAddedListener((p, index, color) -> {
            viewModel.onEpipolarPointAdded(p, index, color, "Point " + index);
        });

        findViewById(R.id.mc_btn_gallery).setOnClickListener(v -> {
            ImageCaptureHelper.launchGalleryPicker(galleryLauncher);
            showComparisonView();
        });

        findViewById(R.id.mc_btn_epipolar).setOnClickListener(v -> {
            if (epipolarView.getVisibility() == View.VISIBLE) {
                showComparisonView();
            } else {
                showEpipolarView();
            }
        });

        findViewById(R.id.mc_btn_save).setOnClickListener(v -> {
            Bitmap bmp = viewModel.getCorrectedImage().getValue();
            if (bmp != null) {
                Uri uri = FileExporter.saveBitmapAsPng(this, bmp, "Corrected_" + System.currentTimeMillis());
                if (uri != null) {
                    Toast.makeText(this, R.string.mc_save_success, Toast.LENGTH_SHORT).show();
                } else {
                    Toast.makeText(this, R.string.mc_save_error, Toast.LENGTH_SHORT).show();
                }
            }
        });

        findViewById(R.id.mc_btn_csv).setOnClickListener(v -> {
            com.example.visionproject.modelocamera.logger.CalibrationSessionLogger.getInstance().endSession(getApplicationContext(), "USER");
            Toast.makeText(this, R.string.mc_csv_save_success, Toast.LENGTH_LONG).show();
        });

        findViewById(R.id.mc_btn_share).setOnClickListener(v -> {
            Bitmap bmp = viewModel.getCorrectedImage().getValue();
            if (bmp != null) {
                Uri uri = FileExporter.saveBitmapAsPng(this, bmp, "Share_" + System.currentTimeMillis());
                if (uri != null) {
                    FileExporter.shareImage(this, uri);
                }
            }
        });

        findViewById(R.id.mc_btn_back).setOnClickListener(v -> finish());
    }

    private void showComparisonView() {
        comparisonView.setVisibility(View.VISIBLE);
        epipolarView.setVisibility(View.GONE);
    }

    private void showEpipolarView() {
        if (viewModel.getOriginalImage().getValue() == null) {
            Toast.makeText(this, "Carregue uma imagem primeiro", Toast.LENGTH_SHORT).show();
            return;
        }
        comparisonView.setVisibility(View.GONE);
        epipolarView.setVisibility(View.VISIBLE);
        Toast.makeText(this, R.string.mc_epipolar_instructions, Toast.LENGTH_LONG).show();
    }
}
