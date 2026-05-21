package com.example.visionproject.calibracao;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import com.example.visionproject.calibracao.strategy.PoseStabilityStrategy;

import org.junit.Test;
import org.opencv.core.CvType;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;

public class PoseStabilityStrategyTest {

    @Test
    public void testStability() {
        try {
            System.loadLibrary(org.opencv.core.Core.NATIVE_LIBRARY_NAME);
        } catch (UnsatisfiedLinkError e) {
            return;
        }

        PoseStabilityStrategy strategy = new PoseStabilityStrategy(3, 2.0);
        
        MatOfPoint2f pts = new MatOfPoint2f(new Point(100, 100), new Point(200, 100));
        
        // Primeiros frames nunca são estáveis (precisa encher o histórico)
        assertFalse(strategy.checkStability(pts));
        assertFalse(strategy.checkStability(pts));
        
        // Terceiro frame com mesma posição deve ser estável
        assertTrue(strategy.checkStability(pts));
        
        // Pequeno movimento ( < 2.0 px) deve continuar estável
        MatOfPoint2f ptsSmallMove = new MatOfPoint2f(new Point(100.5, 100.5), new Point(200.5, 100.5));
        assertTrue(strategy.checkStability(ptsSmallMove));

        // Grande movimento deve quebrar estabilidade
        MatOfPoint2f ptsLargeMove = new MatOfPoint2f(new Point(150, 150), new Point(250, 150));
        assertFalse(strategy.checkStability(ptsLargeMove));
        
        pts.release();
        ptsSmallMove.release();
        ptsLargeMove.release();
    }
}
