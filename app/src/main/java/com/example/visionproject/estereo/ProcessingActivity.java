package com.example.visionproject.estereo;

import android.content.Intent;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.lifecycle.ViewModelProvider;

import com.example.visionproject.R;
import com.example.visionproject.estereo.model.StereoPair;
import com.example.visionproject.estereo.repository.StereoPairRepository;
import com.example.visionproject.estereo.viewmodel.ProcessingViewModel;

public class ProcessingActivity extends AppCompatActivity {

    private ProcessingViewModel viewModel;
    private TextView tvStatus;
    private ProgressBar progressBar;
    private ImageView ivPreview;
    private Button btnNext;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.est_activity_processing);

        tvStatus = findViewById(R.id.tv_processing_status);
        progressBar = findViewById(R.id.pb_processing);
        ivPreview = findViewById(R.id.iv_artifact_preview);
        btnNext = findViewById(R.id.btn_next_depth);

        viewModel = new ViewModelProvider(this).get(ProcessingViewModel.class);

        observeViewModel();

        // Inicia o processamento do par atual no repositório
        StereoPair pair = StereoPairRepository.getInstance().getCurrentPair();
        if (pair == null) {
            Toast.makeText(this, "Erro: Capture um par estereo primeiro.", Toast.LENGTH_LONG).show();
            finish();
            return;
        }
        viewModel.processPair(pair);

        btnNext.setOnClickListener(v -> {
            startActivity(new Intent(this, DepthActivity.class));
            finish();
        });
    }

    private void observeViewModel() {
        viewModel.getStatusMessage().observe(this, msg -> tvStatus.setText(msg));

        viewModel.getCurrentArtifact().observe(this, bmp -> {
            if (bmp != null) {
                ivPreview.setImageBitmap(bmp);
            }
        });

        viewModel.getState().observe(this, state -> {
            switch (state) {
                case PROCESSING:
                    progressBar.setVisibility(View.VISIBLE);
                    btnNext.setEnabled(false);
                    break;
                case DONE:
                    progressBar.setVisibility(View.GONE);
                    btnNext.setEnabled(true);
                    Toast.makeText(this, "Processamento concluído!", Toast.LENGTH_SHORT).show();
                    break;
                case ERROR:
                    progressBar.setVisibility(View.GONE);
                    btnNext.setEnabled(false);
                    break;
            }
        });
    }
}
