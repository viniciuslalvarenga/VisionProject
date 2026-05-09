package com.example.visionproject;

import android.os.Bundle;
import android.util.Log;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;

import java.util.Locale;

public class ModeloCameraActivity extends AppCompatActivity {
    private static final String TAG = "ModeloCameraActivity";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_modelo_camera);

        TextView tvMatrixK = findViewById(R.id.tv_matrix_k);
        TextView tvDistCoeffs = findViewById(R.id.tv_dist_coeffs);
        TextView tvPointCorr = findViewById(R.id.tv_point_corr);

        // 1. Criar matriz de câmera K (Valores estimados para smartphone 1080p - Calibrados)
        Mat kMat = new Mat(3, 3, CvType.CV_64F);
        double fx = 1408.5, fy = 1410.2, cx = 955.8, cy = 538.1;
        kMat.put(0, 0, fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0);

        tvMatrixK.setText(getString(R.string.matrix_k_format, fx, cx, fy, cy));

        // 2. Coeficientes de distorção (Exemplo: k1, k2, p1, p2, k3)
        MatOfDouble distCoeffs = new MatOfDouble(0.12, -0.08, 0.001, 0.0005, 0.0);
        tvDistCoeffs.setText(R.string.dist_coeffs_value);

        // 3. Exemplo de correção de ponto (undistortPoints)
        Point distortedPoint = new Point(1200, 800);
        MatOfPoint2f distortedPts = new MatOfPoint2f(distortedPoint);
        MatOfPoint2f undistortedPts = new MatOfPoint2f();

        Calib3d.undistortPoints(distortedPts, undistortedPts, kMat, distCoeffs);

        Point p = undistortedPts.toArray()[0];
        
        // Importante: undistortPoints retorna coordenadas normalizadas se P ou R não forem passados.
        // Para voltar para pixels: x' = x*fx + cx, y' = y*fy + cy
        double correctedX = p.x * fx + cx;
        double correctedY = p.y * fy + cy;

        tvPointCorr.setText(getString(R.string.point_corrected, correctedX, correctedY));
        Log.d(TAG, "Ponto original: " + distortedPoint.toString());
        Log.d(TAG, "Ponto corrigido: (" + correctedX + ", " + correctedY + ")");

        findViewById(R.id.btn_back).setOnClickListener(v -> finish());

        // Liberar memória OpenCV
        kMat.release();
        distCoeffs.release();
        distortedPts.release();
        undistortedPts.release();
    }
}
