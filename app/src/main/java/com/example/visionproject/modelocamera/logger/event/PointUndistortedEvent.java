package com.example.visionproject.modelocamera.logger.event;

import com.example.visionproject.modelocamera.logger.SessionEvent;
import com.example.visionproject.shared.csv.CsvFormatter;

import org.opencv.core.Point;

import java.util.LinkedHashMap;
import java.util.Map;

public class PointUndistortedEvent extends SessionEvent {
    public final Point original;
    public final Point corrected;
    public final double distFromCenter;

    public PointUndistortedEvent(Point original, Point corrected, double distFromCenter) {
        super("POINT_UNDISTORT");
        this.original = original;
        this.corrected = corrected;
        this.distFromCenter = distFromCenter;
    }

    @Override
    public Map<String, String> toCsvFields(CsvFormatter fmt) {
        Map<String, String> m = new LinkedHashMap<>();
        m.put("x_orig", fmt.fmtDouble(original.x, 2));
        m.put("y_orig", fmt.fmtDouble(original.y, 2));
        m.put("x_corr", fmt.fmtDouble(corrected.x, 2));
        m.put("y_corr", fmt.fmtDouble(corrected.y, 2));
        m.put("delta_x", fmt.fmtDouble(corrected.x - original.x, 2));
        m.put("delta_y", fmt.fmtDouble(corrected.y - original.y, 2));
        m.put("dist_from_center", fmt.fmtDouble(distFromCenter, 1));
        return m;
    }
}
