package com.example.visionproject.estereo.model;

import java.io.File;
import java.io.Serializable;

public class StereoPair implements Serializable {
    public final File leftFile;
    public final File rightFile;
    public final float baselineMm;
    public final String sessionId;

    public StereoPair(File leftFile, File rightFile, float baselineMm, String sessionId) {
        this.leftFile = leftFile;
        this.rightFile = rightFile;
        this.baselineMm = baselineMm;
        this.sessionId = sessionId;
    }
}
