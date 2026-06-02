package com.example.visionproject.estereo;

public enum EstereoState {
    IDLE,
    AWAIT_LEFT,
    AWAIT_RIGHT,
    PAIR_READY,
    PROCESSING,
    DONE,
    ERROR
}
