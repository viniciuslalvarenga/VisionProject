package com.example.visionproject.shared.csv;

import org.opencv.core.Point;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

/**
 * Utilitário de formatação padrão para CSV (RFC 4180).
 */
public class CsvFormatter {

    public static String fmtDouble(double v, int decimals) {
        return String.format(Locale.US, "%." + decimals + "f", v);
    }

    public static String fmtPoint(Point p, int decimals) {
        if (p == null) return "";
        return fmtDouble(p.x, decimals); // Chamado duas vezes para x e y
    }

    public static String escape(String field) {
        if (field == null) return "";
        if (field.contains(",") || field.contains("\"") || field.contains("\n")) {
            return "\"" + field.replace("\"", "\"\"") + "\"";
        }
        return field;
    }

    public static String fmtTimestampIso(long ms) {
        SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss", Locale.US);
        return sdf.format(new Date(ms));
    }

    public static String fmtColor(int rgb) {
        return String.format("#%06X", (0xFFFFFF & rgb));
    }
}
