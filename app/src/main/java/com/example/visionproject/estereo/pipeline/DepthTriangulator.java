package com.example.visionproject.estereo.pipeline;

import org.opencv.core.CvType;
import org.opencv.core.Mat;

public class DepthTriangulator {
    /**
     * Calcula o mapa de profundidade Z = (f * B) / d
     * @param disp32f Mapa de disparidade em 32-bit float
     * @param fxPixels Distância focal em pixels (da matriz K)
     * @param baselineMeters Baseline em metros
     * @return Mat de profundidade (CV_32F) em metros
     */
    public static Mat triangulate(Mat disp32f, double fxPixels, double baselineMeters) {
        Mat depth = new Mat(disp32f.size(), CvType.CV_32F);
        int rows = disp32f.rows();
        int cols = disp32f.cols();
        
        float[] dispData = new float[rows * cols];
        float[] depthData = new float[rows * cols];
        
        disp32f.get(0, 0, dispData);
        
        double fb = fxPixels * baselineMeters;
        
        for (int i = 0; i < dispData.length; i++) {
            float d = dispData[i];
            float dAbs = Math.abs(d);
            if (dAbs > 0.1f) { // Evita divisão por zero e ruído excessivo
                depthData[i] = (float) (fb / dAbs);
            } else {
                depthData[i] = 0f;
            }
        }
        
        depth.put(0, 0, depthData);
        return depth;
    }
}
