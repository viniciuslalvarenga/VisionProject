package com.example.visionproject.calibracao.repository;

import com.example.visionproject.calibracao.model.CalibrationFrame;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Repositório Singleton para armazenar os frames coletados para calibração.
 */
public class CalibrationFramesRepository {
    private static CalibrationFramesRepository instance;
    private final List<CalibrationFrame> frames = Collections.synchronizedList(new ArrayList<>());

    private CalibrationFramesRepository() {}

    public static synchronized CalibrationFramesRepository getInstance() {
        if (instance == null) {
            instance = new CalibrationFramesRepository();
        }
        return instance;
    }

    public void addFrame(CalibrationFrame frame) {
        frames.add(frame);
    }

    public void removeAt(int index) {
        if (index >= 0 && index < frames.size()) {
            CalibrationFrame frame = frames.remove(index);
            if (frame != null) frame.release();
        }
    }

    public void clear() {
        for (CalibrationFrame f : frames) {
            f.release();
        }
        frames.clear();
    }

    public int size() {
        return frames.size();
    }

    public List<CalibrationFrame> getAll() {
        return new ArrayList<>(frames);
    }
}
