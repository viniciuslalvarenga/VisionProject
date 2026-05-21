package com.example.visionproject.calibracao;

import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import com.example.visionproject.calibracao.strategy.BlurDetectionStrategy;

import org.junit.Test;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

public class BlurDetectionStrategyTest {

    @Test
    public void testGetScore() {
        try {
            System.loadLibrary(org.opencv.core.Core.NATIVE_LIBRARY_NAME);
        } catch (UnsatisfiedLinkError e) {
            return;
        }

        BlurDetectionStrategy strategy = new BlurDetectionStrategy();
        
        // Imagem uniforme (cinza) deve ter score baixo
        Mat gray = new Mat(100, 100, CvType.CV_8UC1, new org.opencv.core.Scalar(128));
        double scoreUniform = strategy.getScore(gray);
        assertTrue(scoreUniform < 10.0);

        // Imagem com alto contraste (xadrez) deve ter score alto
        for (int i = 0; i < 100; i++) {
            for (int j = 0; j < 100; j++) {
                if ((i/10 + j/10) % 2 == 0) {
                    gray.put(i, j, 0);
                } else {
                    gray.put(i, j, 255);
                }
            }
        }
        double scorePattern = strategy.getScore(gray);
        assertTrue(scorePattern > 1000.0);
        
        gray.release();
    }
}
