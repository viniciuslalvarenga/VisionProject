package com.example.visionproject.vio;

import android.graphics.Bitmap;
import android.os.Bundle;
import android.text.InputType;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.lifecycle.ViewModelProvider;

import com.example.visionproject.R;
import com.example.visionproject.vio.ui.DisparityClickView;
import com.example.visionproject.vio.viewmodel.VioProcessingViewModel;

public class VioProcessingActivity extends AppCompatActivity {
    private VioProcessingViewModel viewModel;
    private DisparityClickView disparityView;
    private TextView tvStatus, tvMedianZ;
    private Button btnRun, btnExportPly, btnCompareZ;

    private double lastQueriedZ = Double.NaN;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.vio_activity_processing);

        viewModel = new ViewModelProvider(this).get(VioProcessingViewModel.class);

        tvStatus    = findViewById(R.id.tv_vio_proc_status);
        tvMedianZ   = findViewById(R.id.tv_vio_median_z);
        disparityView = findViewById(R.id.vio_disparity_click_view);
        btnRun      = findViewById(R.id.btn_vio_run_pipeline);
        btnExportPly = findViewById(R.id.btn_vio_export_ply);
        btnCompareZ = findViewById(R.id.btn_vio_compare_z);

        disparityView.setDepthClickListener((z, px, py) -> {
            lastQueriedZ = z;
            tvMedianZ.setText(String.format(java.util.Locale.US,
                    "Z clicado: %.3f m  Mediana: %.3f m", z,
                    viewModel.getMedianZ().getValue() != null ?
                    viewModel.getMedianZ().getValue() : Double.NaN));
        });

        observeViewModel();
        setupButtons();
    }

    private void observeViewModel() {
        viewModel.getState().observe(this, s -> {
            tvStatus.setText("Estado: " + s.name());
            boolean done = s == VioState.DONE;
            btnExportPly.setEnabled(done);
            btnCompareZ.setEnabled(done);
        });
        viewModel.getStatus().observe(this, tvStatus::setText);
        viewModel.getArtifact().observe(this, bmp -> {
            if (bmp != null) {
                disparityView.setDisparityBitmap(bmp);
                disparityView.setXyzMat(viewModel.getXyzMat());
            }
        });
        viewModel.getMedianZ().observe(this, z -> {
            if (!Double.isNaN(z)) {
                tvMedianZ.setText(String.format(java.util.Locale.US, "Mediana Z: %.3f m", z));
            }
        });
    }

    private void setupButtons() {
        btnRun.setOnClickListener(v -> {
            btnRun.setEnabled(false);
            viewModel.runPipeline();
        });

        btnExportPly.setEnabled(false);
        btnExportPly.setOnClickListener(v -> {
            String name = viewModel.exportPly();
            if (name != null) {
                Toast.makeText(this, "PLY exportado: " + name, Toast.LENGTH_LONG).show();
            } else {
                Toast.makeText(this, "Erro ao exportar PLY", Toast.LENGTH_SHORT).show();
            }
        });

        btnCompareZ.setEnabled(false);
        btnCompareZ.setOnClickListener(v -> showCompareZDialog());
    }

    private void showCompareZDialog() {
        EditText input = new EditText(this);
        input.setInputType(InputType.TYPE_CLASS_NUMBER | InputType.TYPE_NUMBER_FLAG_DECIMAL);
        input.setHint("Ex: 1.50");
        new AlertDialog.Builder(this)
                .setTitle("Comparar com fita métrica")
                .setMessage(String.format(java.util.Locale.US,
                        "Z estimado: %.3f m\nDigite o Z real (metros):", lastQueriedZ))
                .setView(input)
                .setPositiveButton("Calcular", (d, w) -> {
                    try {
                        double zReal = Double.parseDouble(input.getText().toString().replace(',', '.'));
                        if (zReal <= 0) throw new NumberFormatException();
                        double errPct = Math.abs(lastQueriedZ - zReal) / zReal * 100.0;
                        viewModel.logScaleError(lastQueriedZ, zReal);
                        Toast.makeText(this,
                                String.format(java.util.Locale.US,
                                        "Z_est=%.3fm Z_real=%.3fm Erro=%.1f%%",
                                        lastQueriedZ, zReal, errPct),
                                Toast.LENGTH_LONG).show();
                    } catch (NumberFormatException ex) {
                        Toast.makeText(this, "Valor inválido", Toast.LENGTH_SHORT).show();
                    }
                })
                .setNegativeButton("Cancelar", null)
                .show();
    }
}
