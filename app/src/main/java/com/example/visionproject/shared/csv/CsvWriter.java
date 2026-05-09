package com.example.visionproject.shared.csv;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.util.Map;

/**
 * Wrapper thread-safe para escrita de arquivos CSV.
 */
public class CsvWriter {
    private final BufferedWriter writer;
    private final String[] columns;

    public CsvWriter(OutputStream out, String[] columns) {
        this.writer = new BufferedWriter(new OutputStreamWriter(out, StandardCharsets.UTF_8));
        this.columns = columns;
    }

    public synchronized void writeHeader() throws IOException {
        for (int i = 0; i < columns.length; i++) {
            writer.write(columns[i]);
            if (i < columns.length - 1) writer.write(",");
        }
        writer.write("\r\n");
        writer.flush();
    }

    public synchronized void writeRow(Map<String, String> row) throws IOException {
        for (int i = 0; i < columns.length; i++) {
            String value = row.get(columns[i]);
            writer.write(value != null ? value : "");
            if (i < columns.length - 1) writer.write(",");
        }
        writer.write("\r\n");
        writer.flush();
    }

    public synchronized void close() throws IOException {
        writer.flush();
        writer.close();
    }
}
