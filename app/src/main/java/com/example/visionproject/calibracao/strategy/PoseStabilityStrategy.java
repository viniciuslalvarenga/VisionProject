package com.example.visionproject.calibracao.strategy;

import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;

import java.util.LinkedList;
import java.util.Queue;

/**
 * Estratégia para verificar a estabilidade da pose do padrão.
 * Compara o movimento dos cantos detectados entre frames consecutivos na janela.
 */
public class PoseStabilityStrategy {
    private final int historySize;
    private final double threshold;
    private final Queue<MatOfPoint2f> history = new LinkedList<>();

    public PoseStabilityStrategy() {
        this(5, 2.0); // 5 frames, shift médio < 2.0px
    }

    public PoseStabilityStrategy(int historySize, double threshold) {
        this.historySize = historySize;
        this.threshold = threshold;
    }

    /**
     * @deprecated Use checkStability(MatOfPoint2f) em vez deste. Mantido para compatibilidade.
     */
    @Deprecated
    public boolean isStable(MatOfPoint2f currentCorners) {
        return checkStability(currentCorners);
    }

    public boolean checkStability(MatOfPoint2f currentCorners) {
        if (currentCorners == null) {
            reset();
            return false;
        }

        // clone defensivo
        MatOfPoint2f clone = new MatOfPoint2f();
        currentCorners.copyTo(clone);

        // Mantém janela de tamanho fixo
        if (history.size() >= historySize) {
            MatOfPoint2f removed = history.poll();
            if (removed != null) removed.release();
        }
        history.add(clone);

        // Precisa pelo menos N frames para julgar estabilidade
        if (history.size() < historySize) return false;

        // Verifica shift MÁXIMO entre pares consecutivos da janela
        MatOfPoint2f[] arr = history.toArray(new MatOfPoint2f[0]);
        double maxShift = 0;
        for (int i = 1; i < arr.length; i++) {
            double s = calculateMeanShift(arr[i-1], arr[i]);
            if (s > maxShift) maxShift = s;
        }
        return maxShift < threshold;
    }

    public double getCurrentStabilityScore() {
        if (history.size() < 2) return Double.NaN;
        MatOfPoint2f[] arr = history.toArray(new MatOfPoint2f[0]);
        double maxShift = 0;
        for (int i = 1; i < arr.length; i++) {
            double s = calculateMeanShift(arr[i-1], arr[i]);
            if (s > maxShift) maxShift = s;
        }
        return maxShift;
    }

    private double calculateMeanShift(MatOfPoint2f pts1, MatOfPoint2f pts2) {
        if (pts1.rows() != pts2.rows()) return Double.MAX_VALUE;
        
        Point[] p1 = pts1.toArray();
        Point[] p2 = pts2.toArray();
        double totalDist = 0;
        for (int i = 0; i < p1.length; i++) {
            totalDist += Math.sqrt(Math.pow(p1[i].x - p2[i].x, 2) + Math.pow(p1[i].y - p2[i].y, 2));
        }
        return totalDist / p1.length;
    }

    public void reset() {
        for (MatOfPoint2f m : history) {
            m.release();
        }
        history.clear();
    }
}
