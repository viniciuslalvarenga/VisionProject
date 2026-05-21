package com.example.visionproject.calibracao;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

import com.example.visionproject.calibracao.factory.ObjectPointsFactory;
import com.example.visionproject.calibracao.model.ChessboardSpec;

import org.junit.Test;
import org.opencv.core.Mat;

public class ObjectPointsFactoryTest {

    @Test
    public void testCreateForPattern() {
        try {
            System.loadLibrary(org.opencv.core.Core.NATIVE_LIBRARY_NAME);
        } catch (UnsatisfiedLinkError e) {
            return;
        }

        ChessboardSpec spec = new ChessboardSpec(9, 6, 25.0f);
        Mat objPoints = ObjectPointsFactory.createForPattern(spec);
        
        assertNotNull(objPoints);
        assertEquals(54, objPoints.rows());
        
        double[] last = objPoints.get(53, 0);
        assertEquals(0.0, last[2], 0.001); // Z=0
    }
}
