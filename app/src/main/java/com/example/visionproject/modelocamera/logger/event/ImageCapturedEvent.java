package com.example.visionproject.modelocamera.logger.event;

import com.example.visionproject.modelocamera.logger.SessionEvent;
import com.example.visionproject.shared.csv.CsvFormatter;

import java.util.LinkedHashMap;
import java.util.Map;

public class ImageCapturedEvent extends SessionEvent {
    public final String filename;
    public final int width;
    public final int height;
    public final String source;

    public ImageCapturedEvent(String filename, int width, int height, String source) {
        super("IMAGE_CAPTURED");
        this.filename = filename;
        this.width = width;
        this.height = height;
        this.source = source;
    }

    @Override
    public Map<String, String> toCsvFields(CsvFormatter fmt) {
        Map<String, String> m = new LinkedHashMap<>();
        m.put("image_filename", filename);
        m.put("res_w", String.valueOf(width));
        m.put("res_h", String.valueOf(height));
        m.put("image_source", source);
        return m;
    }
}
