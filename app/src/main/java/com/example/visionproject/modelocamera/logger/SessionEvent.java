package com.example.visionproject.modelocamera.logger;

import com.example.visionproject.shared.csv.CsvFormatter;

import java.util.Map;

/**
 * Base para todos os eventos de sessão de calibração.
 */
public abstract class SessionEvent {
    public final long timestampMs;
    public final String eventType;

    protected SessionEvent(String eventType) {
        this.timestampMs = System.currentTimeMillis();
        this.eventType = eventType;
    }

    /**
     * Retorna os campos específicos deste evento para o CSV.
     */
    public abstract Map<String, String> toCsvFields(CsvFormatter fmt);
}
