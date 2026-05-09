package com.example.visionproject.modelocamera.logger.event;

import com.example.visionproject.modelocamera.logger.SessionEvent;
import com.example.visionproject.shared.csv.CsvFormatter;

import java.util.LinkedHashMap;
import java.util.Map;

public class SessionClosedEvent extends SessionEvent {
    public final int totalEvents;
    public final String reason;

    public SessionClosedEvent(int totalEvents, String reason) {
        super("SESSION_CLOSED");
        this.totalEvents = totalEvents;
        this.reason = reason;
    }

    @Override
    public Map<String, String> toCsvFields(CsvFormatter fmt) {
        Map<String, String> m = new LinkedHashMap<>();
        m.put("notes", "Session closed. Total events: " + totalEvents + ". Reason: " + reason);
        return m;
    }
}
