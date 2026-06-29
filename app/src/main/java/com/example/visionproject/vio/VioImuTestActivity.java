package com.example.visionproject.vio;

import android.hardware.SensorManager;
import android.os.Bundle;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.lifecycle.ViewModelProvider;

import com.example.visionproject.R;
import com.example.visionproject.vio.repository.VioCsvLogger;
import com.example.visionproject.vio.sensor.SensorLogger;
import com.example.visionproject.vio.ui.ImuDriftPlotView;
import com.example.visionproject.vio.viewmodel.VioImuTestViewModel;

public class VioImuTestActivity extends AppCompatActivity {
    private VioImuTestViewModel viewModel;
    private SensorLogger sensorLogger;
    private SensorManager sensorManager;

    private ImuDriftPlotView plotView;
    private TextView tvResult, tvInstruction;
    private Button btnStart, btnStop, btnSave;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.vio_activity_imu_test);

        viewModel = new ViewModelProvider(this).get(VioImuTestViewModel.class);
        sensorManager = (SensorManager) getSystemService(SENSOR_SERVICE);

        plotView     = findViewById(R.id.vio_imu_plot);
        tvResult     = findViewById(R.id.tv_imu_result);
        tvInstruction= findViewById(R.id.tv_imu_instruction);
        btnStart     = findViewById(R.id.btn_imu_start);
        btnStop      = findViewById(R.id.btn_imu_stop);
        btnSave      = findViewById(R.id.btn_imu_save);

        setupSensorLogger();
        observeViewModel();
        setupButtons();
    }

    private void setupSensorLogger() {
        sensorLogger = new SensorLogger();
        sensorLogger.setListener(new SensorLogger.Listener() {
            @Override public void onAccel(SensorLogger.Sample s) {
                viewModel.onAccelSample(s.tNs, s.v);
            }
            @Override public void onGyro(SensorLogger.Sample s) {}
            @Override public void onRotation(SensorLogger.Sample s) {
                viewModel.onRotation(s.v);
            }
        });
    }

    private void observeViewModel() {
        viewModel.isRecording().observe(this, rec -> {
            btnStart.setEnabled(!rec);
            btnStop.setEnabled(rec);
            tvInstruction.setText(rec
                    ? "Mova o phone 1m para FRENTE e depois VOLTE ao ponto inicial."
                    : "Pressione INICIAR para começar o experimento de drift IMU.");
        });

        viewModel.getResult().observe(this, tvResult::setText);

        viewModel.getTraj1().observe(this, t1 -> {
            plotView.setData(t1,
                    viewModel.getTraj2().getValue(),
                    viewModel.getErrors1().getValue(),
                    viewModel.getErrors2().getValue());
        });
    }

    private void setupButtons() {
        btnStart.setOnClickListener(v -> viewModel.startRecording());
        btnStop.setOnClickListener(v -> viewModel.stopRecording());
        btnSave.setOnClickListener(v -> {
            VioCsvLogger.getInstance().saveSession(this);
            Toast.makeText(this, "Dados IMU salvos no CSV.", Toast.LENGTH_SHORT).show();
        });
        btnStop.setEnabled(false);
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (sensorLogger != null) sensorLogger.start(sensorManager);
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (sensorLogger != null) sensorLogger.stop(sensorManager);
    }
}
