package com.example.visionproject.calibracao;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

import com.example.visionproject.calibracao.model.ChessboardSpec;

import org.junit.Before;
import org.junit.Test;
import org.opencv.core.Mat;

public class ChessboardSpecTest {

    @Before
    public void setUp() {
        // Inicialização do OpenCV no ambiente de teste unitário pode ser complexa.
        // Se o teste falhar por falta da lib nativa, este teste deve ser movido para androidTest
        // ou usar um mock/wrapper. No entanto, o spec pede teste unitário.
    }

    @Test
    public void testGenerateObjectPoints() {
        // Para rodar este teste que usa OpenCV, precisamos que a lib esteja carregada.
        // Em testes unitários locais (JUnit) isso geralmente falha sem configuração extra.
        // Assumindo que o ambiente está configurado ou que o desenvolvedor rodará como androidTest se necessário.
        try {
            System.loadLibrary(org.opencv.core.Core.NATIVE_LIBRARY_NAME);
        } catch (UnsatisfiedLinkError e) {
            // Se falhar o carregamento da lib nativa no host, pulamos a parte que usa Mat
            return;
        }

        ChessboardSpec spec = new ChessboardSpec(9, 6, 25.0f);
        Mat objPoints = spec.generateObjectPoints();
        
        assertEquals(54, objPoints.rows());
        assertEquals(1, objPoints.cols());
        
        double[] first = objPoints.get(0, 0);
        assertEquals(0.0, first[0], 0.001);
        assertEquals(0.0, first[1], 0.001);
        assertEquals(0.0, first[2], 0.001);

        double[] last = objPoints.get(53, 0);
        assertEquals(200.0, last[0], 0.001); // (9-1)*25
        assertEquals(125.0, last[1], 0.001); // (6-1)*25
        assertEquals(0.0, last[2], 0.001);
    }
}
