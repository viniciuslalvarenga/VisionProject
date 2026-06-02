package com.example.visionproject.estereo;

import android.content.Intent;
import android.os.Bundle;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.example.visionproject.R;
import com.example.visionproject.calibracao.model.CalibrationResult;
import com.example.visionproject.calibracao.repository.CalibrationJsonStore;
import com.example.visionproject.estereo.repository.StereoPairRepository;

public class EstereoMainActivity extends AppCompatActivity {

    private TextView tvStatusCalib;
    private Button btnCapture, btnProcess, btnDepth, btnGoCalib;
    private CalibrationResult calibrationResult;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.est_activity_main);

        tvStatusCalib = findViewById(R.id.tv_status_calib);
        btnCapture = findViewById(R.id.btn_capture_pair);
        btnProcess = findViewById(R.id.btn_process_pair);
        btnDepth = findViewById(R.id.btn_measure_depth);
        btnGoCalib = findViewById(R.id.btn_go_calib);

        checkCalibration();

        btnCapture.setOnClickListener(v -> startActivity(new Intent(this, CaptureActivity.class)));
        
        btnProcess.setOnClickListener(v -> {
            if (StereoPairRepository.getInstance().getCurrentPair() != null) {
                startActivity(new Intent(this, ProcessingActivity.class));
            } else {
                Toast.makeText(this, "Capture um par primeiro", Toast.LENGTH_SHORT).show();
            }
        });

        btnDepth.setOnClickListener(v -> {
             // Será implementado após T6/T7
             Toast.makeText(this, "Processamento necessário primeiro", Toast.LENGTH_SHORT).show();
        });

        btnGoCalib.setOnClickListener(v -> {
            try {
                Class<?> calibActivity = Class.forName("com.example.visionproject.calibracao.CalibrationActivity");
                startActivity(new Intent(this, calibActivity));
            } catch (ClassNotFoundException e) {
                Toast.makeText(this, "Módulo de calibração não encontrado", Toast.LENGTH_SHORT).show();
            }
        });
    }

    @Override
    protected void onResume() {
        super.onResume();
        checkCalibration();
        updateButtons();
    }

    private void checkCalibration() {
        calibrationResult = CalibrationJsonStore.load(this);
        if (calibrationResult != null) {
            tvStatusCalib.setText(getString(R.string.est_status_calib_ok, 
                    calibrationResult.getCameraMatrix().get(0, 0)[0], 
                    calibrationResult.getRms()));
            tvStatusCalib.setTextColor(getResources().getColor(android.R.color.holo_green_dark));
            btnGoCalib.setVisibility(Button.GONE);
            btnCapture.setEnabled(true);
        } else {
            tvStatusCalib.setText(R.string.est_status_calib_error);
            tvStatusCalib.setTextColor(getResources().getColor(android.R.color.holo_red_dark));
            btnGoCalib.setVisibility(Button.VISIBLE);
            btnCapture.setEnabled(false);
        }
    }

    private void updateButtons() {
        boolean hasPair = StereoPairRepository.getInstance().getCurrentPair() != null;
        btnProcess.setEnabled(hasPair);
        // Depth depende de processamento concluído (implementaremos flag no repo depois)
        btnDepth.setEnabled(false); 
    }
}
