package com.example.visionproject.calibracao.factory;

import com.example.visionproject.calibracao.model.PatternSpec;
import org.opencv.core.Mat;

/**
 * Factory para criação de coordenadas de pontos do objeto (3D).
 */
public class ObjectPointsFactory {
    public static Mat createForPattern(PatternSpec p) {
        return p.generateObjectPoints();
    }
}
