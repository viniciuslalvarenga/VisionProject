package com.example.visionproject.calibracao.model;

/**
 * Estados do pipeline de calibração.
 */
public enum CalibrationState {
    IDLE,                  // App aberto, antes de iniciar
    DETECTING,             // Procurando tabuleiro no preview
    POSE_STABLE,           // Tabuleiro detectado e parado, pronto pra capturar
    CAPTURED,              // Frame acabou de ser adicionado
    READY_TO_CALIBRATE,    // >=15 frames coletados, botão "Calibrar" habilitado
    CALIBRATING,           // calibrateCamera() rodando
    DONE,                  // Resultado disponível
    ERROR
}
