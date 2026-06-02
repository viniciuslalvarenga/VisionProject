package com.example.visionproject.estereo;

import android.annotation.SuppressLint;
import android.app.AlertDialog;
import android.graphics.Matrix;
import android.graphics.drawable.Drawable;
import android.os.Bundle;
import android.text.InputType;
import android.view.MotionEvent;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.lifecycle.ViewModelProvider;

import com.example.visionproject.R;
import com.example.visionproject.estereo.repository.StereoCsvLogger;
import com.example.visionproject.estereo.viewmodel.DepthViewModel;

public class DepthActivity extends AppCompatActivity {

    private DepthViewModel viewModel;
    private ImageView ivDisparity;
    private TextView tvDepthInfo;
    private Button btnExportPly, btnFinish, btnCompareRef;

    @SuppressLint("ClickableViewAccessibility")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.est_activity_depth);

        ivDisparity = findViewById(R.id.iv_disparity_map);
        tvDepthInfo = findViewById(R.id.tv_depth_info);
        btnExportPly = findViewById(R.id.btn_export_ply);
        btnFinish = findViewById(R.id.btn_finish_estereo);
        btnCompareRef = findViewById(R.id.btn_compare_ref);

        viewModel = new ViewModelProvider(this).get(DepthViewModel.class);
        viewModel.init();

        observeViewModel();

        ivDisparity.setOnTouchListener((v, event) -> {
            if (event.getAction() == MotionEvent.ACTION_DOWN) {
                float[] mapped = mapViewToBitmap(ivDisparity, event.getX(), event.getY());
                if (mapped != null) {
                    viewModel.onMapClick((int) mapped[0], (int) mapped[1]);
                }
            }
            return true;
        });

        btnExportPly.setOnClickListener(v -> {
            viewModel.exportPly();
            Toast.makeText(this, "Exportando nuvem de pontos...", Toast.LENGTH_SHORT).show();
        });

        if (btnCompareRef != null) {
            btnCompareRef.setOnClickListener(v -> showReferenceCompareDialog());
        }

        btnFinish.setOnClickListener(v -> {
            // Salvar CSV com todos os eventos antes de sair
            StereoCsvLogger.getInstance().saveSession(this);
            // Aguardar um pouco para garantir que o save iniciou
            v.postDelayed(this::finish, 500);
        });
    }

    /** T8 - dialog para digitar a distancia real (fita metrica) e comparar com Z. */
    private void showReferenceCompareDialog() {
        final EditText input = new EditText(this);
        input.setInputType(InputType.TYPE_CLASS_NUMBER | InputType.TYPE_NUMBER_FLAG_DECIMAL);
        input.setHint("ex: 1.20 (metros)");

        new AlertDialog.Builder(this)
                .setTitle("Comparar com fita metrica")
                .setMessage("Digite a distancia REAL (em metros) do ultimo ponto que voce tocou no mapa.")
                .setView(input)
                .setPositiveButton("Comparar", (d, w) -> {
                    try {
                        double refM = Double.parseDouble(input.getText().toString());
                        viewModel.compareWithReference(refM);
                    } catch (NumberFormatException ex) {
                        Toast.makeText(this, "Valor invalido", Toast.LENGTH_SHORT).show();
                    }
                })
                .setNegativeButton("Cancelar", null)
                .show();
    }

    private void observeViewModel() {
        viewModel.getDisparityBitmap().observe(this, bmp -> {
            if (bmp != null) ivDisparity.setImageBitmap(bmp);
        });
        viewModel.getDepthInfo().observe(this, info -> {
            tvDepthInfo.setText(info);
            // Mostrar mensagens importantes como Toast (sucesso, erro, e resultados)
            if (info != null && !info.equals("Toque no mapa para medir")) {
                if (info.startsWith("Exportado:") || info.startsWith("Erro") || 
                    info.startsWith("Medido:") || info.startsWith("Pixel ") ||
                    info.startsWith("Ponto sem")) {
                    Toast.makeText(this, info, Toast.LENGTH_LONG).show();
                }
            }
        });
    }

    private float[] mapViewToBitmap(ImageView imageView, float x, float y) {
        Drawable drawable = imageView.getDrawable();
        if (drawable == null) return null;

        Matrix inverse = new Matrix();
        if (!imageView.getImageMatrix().invert(inverse)) return null;

        float[] pts = new float[]{x, y};
        inverse.mapPoints(pts);

        int bitmapWidth = drawable.getIntrinsicWidth();
        int bitmapHeight = drawable.getIntrinsicHeight();
        if (pts[0] < 0 || pts[1] < 0 || pts[0] > bitmapWidth || pts[1] > bitmapHeight) {
            return null;
        }
        return pts;
    }
}
