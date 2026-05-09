package com.example.visionproject.modelocamera.logger.event;

import com.example.visionproject.modelocamera.logger.SessionEvent;
import com.example.visionproject.shared.csv.CsvFormatter;

import java.util.LinkedHashMap;
import java.util.Map;

public class ImageUndistortedEvent extends SessionEvent {
    public final String origFilename;
    public final String corrFilename;
    public final int width;
    public final int height;
    public final long elapsedMs;

    public ImageUndistortedEvent(String origFilename, String corrFilename, int width, int height, long elapsedMs) {
        super("IMAGE_UNDISTORTED");
        this.origFilename = origFilename;
        this.corrFilename = corrFilename;
        this.width = width;
        this.height = height;
        this.elapsedMs = elapsedMs;
    }

    @Override
    public Map<String, String> toCsvFields(CsvFormatter fmt) {
        Map<String, String> m = new LinkedHashMap<>();
        m.put("image_filename", corrFilename);
        m.put("res_w", String.valueOf(width));
        m.put("res_h", String.valueOf(height));
        m.put("elapsed_ms", String.valueOf(elapsedMs));
        m.put("notes", "Source: " + origFilename);
        return m;
    }
}
