package com.example.visionproject.modelocamera.logger.event;

import com.example.visionproject.modelocamera.logger.SessionEvent;
import com.example.visionproject.shared.csv.CsvFormatter;

import org.opencv.core.Point;

import java.util.LinkedHashMap;
import java.util.Map;

public class EpipolarPointAddedEvent extends SessionEvent {
    public final Point coords;
    public final int index;
    public final int colorRgb;
    public final String label;

    public EpipolarPointAddedEvent(Point coords, int index, int colorRgb, String label) {
        super("EPIPOLAR_POINT");
        this.coords = coords;
        this.index = index;
        this.colorRgb = colorRgb;
        this.label = label;
    }

    @Override
    public Map<String, String> toCsvFields(CsvFormatter fmt) {
        Map<String, String> m = new LinkedHashMap<>();
        m.put("x_orig", fmt.fmtDouble(coords.x, 2));
        m.put("y_orig", fmt.fmtDouble(coords.y, 2));
        m.put("color_hex", fmt.fmtColor(colorRgb));
        m.put("point_index", String.valueOf(index));
        m.put("notes", label);
        return m;
    }
}
