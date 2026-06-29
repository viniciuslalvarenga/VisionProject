package com.example.visionproject.vio;

import android.content.Intent;
import android.os.Bundle;
import android.widget.Button;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;
import androidx.lifecycle.ViewModelProvider;

import com.example.visionproject.R;
import com.example.visionproject.calibracao.CalibrationActivity;
import com.example.visionproject.vio.viewmodel.VioMainViewModel;

public class VioMainActivity extends AppCompatActivity {
    private VioMainViewModel viewModel;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.vio_activity_main);

        viewModel = new ViewModelProvider(this).get(VioMainViewModel.class);

        TextView tvCalib  = findViewById(R.id.tv_vio_calib_status);
        TextView tvStatus = findViewById(R.id.tv_vio_main_status);
        Button   btnCapture   = findViewById(R.id.btn_vio_capture);
        Button   btnImuTest   = findViewById(R.id.btn_vio_imu_test);
        Button   btnGoCalib   = findViewById(R.id.btn_vio_go_calib);

        viewModel.getCalibration().observe(this, cr -> {
            if (cr == null) {
                tvCalib.setText("Calibração: NÃO ENCONTRADA");
                btnCapture.setEnabled(false);
                btnGoCalib.setEnabled(true);
            } else {
                tvCalib.setText(String.format(java.util.Locale.US,
                        "Calibração: OK  fx=%.1f  RMS=%.3f", cr.getFx(), cr.getRms()));
                btnCapture.setEnabled(true);
                btnGoCalib.setEnabled(false);
            }
        });

        viewModel.getStatusMessage().observe(this, msg -> {
            if (tvStatus != null) tvStatus.setText(msg);
        });

        btnCapture.setOnClickListener(v ->
                startActivity(new Intent(this, VioCaptureActivity.class)));
        btnImuTest.setOnClickListener(v ->
                startActivity(new Intent(this, VioImuTestActivity.class)));
        btnGoCalib.setOnClickListener(v ->
                startActivity(new Intent(this, CalibrationActivity.class)));
    }

    @Override
    protected void onResume() {
        super.onResume();
        viewModel.loadCalibration();
    }
}
