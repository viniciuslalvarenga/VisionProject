package com.example.visionproject.estereo.export;

import org.opencv.core.Mat;
import java.io.File;
import java.io.PrintWriter;
import java.util.Locale;

public class PlyExporter {
    /**
     * Exporta um mapa de profundidade para o formato PLY (nuvem de pontos).
     */
    public static void export(Mat depth, double fx, double cx, double cy, File outFile) throws Exception {
        int rows = depth.rows();
        int cols = depth.cols();
        
        // Conta pontos válidos
        int validPoints = 0;
        float[] depthData = new float[rows * cols];
        depth.get(0, 0, depthData);
        for (float d : depthData) {
            if (d > 0 && !Float.isInfinite(d) && !Float.isNaN(d)) {
                validPoints++;
            }
        }

        try (PrintWriter writer = new PrintWriter(outFile)) {
            writer.println("ply");
            writer.println("format ascii 1.0");
            writer.println("element vertex " + validPoints);
            writer.println("property float x");
            writer.println("property float y");
            writer.println("property float z");
            writer.println("end_header");

            for (int y = 0; y < rows; y++) {
                for (int x = 0; x < cols; x++) {
                    float z = depthData[y * cols + x];
                    if (z > 0 && !Float.isInfinite(z) && !Float.isNaN(z)) {
                        // Back-projection: X = (x - cx) * Z / fx
                        double X = (x - cx) * z / fx;
                        double Y = (y - cy) * z / fx;
                        writer.printf(Locale.US, "%.4f %.4f %.4f\n", X, Y, (double)z);
                    }
                }
            }
            writer.flush();
        }
    }
}
