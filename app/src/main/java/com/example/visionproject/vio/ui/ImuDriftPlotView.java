package com.example.visionproject.vio.ui;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.AttributeSet;
import android.view.View;

import com.example.visionproject.vio.model.Pose;

import java.util.ArrayList;
import java.util.List;

public class ImuDriftPlotView extends View {
    private List<Pose> trajectoryWithZupt = new ArrayList<>();
    private List<Pose> trajectoryNoZupt = new ArrayList<>();
    private List<Float> errorsWithZupt = new ArrayList<>();
    private List<Float> errorsNoZupt = new ArrayList<>();

    private final Paint axisPaint = new Paint();
    private final Paint zuPtPaint = new Paint();
    private final Paint noZuPtPaint = new Paint();
    private final Paint textPaint = new Paint();
    private final Paint gridPaint = new Paint();

    public ImuDriftPlotView(Context ctx) { super(ctx); init(); }
    public ImuDriftPlotView(Context ctx, AttributeSet attrs) { super(ctx, attrs); init(); }
    public ImuDriftPlotView(Context ctx, AttributeSet attrs, int def) { super(ctx, attrs, def); init(); }

    private void init() {
        axisPaint.setColor(Color.WHITE); axisPaint.setStrokeWidth(2f); axisPaint.setStyle(Paint.Style.STROKE);
        zuPtPaint.setColor(Color.GREEN); zuPtPaint.setStrokeWidth(3f); zuPtPaint.setStyle(Paint.Style.STROKE); zuPtPaint.setAntiAlias(true);
        noZuPtPaint.setColor(Color.RED); noZuPtPaint.setStrokeWidth(3f); noZuPtPaint.setStyle(Paint.Style.STROKE); noZuPtPaint.setAntiAlias(true);
        textPaint.setColor(Color.WHITE); textPaint.setTextSize(24f); textPaint.setAntiAlias(true);
        gridPaint.setColor(Color.parseColor("#33FFFFFF")); gridPaint.setStrokeWidth(1f);
    }

    public void setData(List<Pose> withZupt, List<Pose> noZupt,
                        List<Float> errWithZupt, List<Float> errNoZupt) {
        this.trajectoryWithZupt = withZupt != null ? withZupt : new ArrayList<>();
        this.trajectoryNoZupt = noZupt != null ? noZupt : new ArrayList<>();
        this.errorsWithZupt = errWithZupt != null ? errWithZupt : new ArrayList<>();
        this.errorsNoZupt = errNoZupt != null ? errNoZupt : new ArrayList<>();
        invalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        canvas.drawColor(Color.parseColor("#1A1A2E"));
        int w = getWidth(), h = getHeight();
        int halfH = h / 2;

        // Top half: 2D trajectory (XY plane)
        drawTrajectory(canvas, 0, 0, w, halfH);
        // Bottom half: error vs time
        drawErrorCurves(canvas, 0, halfH, w, halfH);
    }

    private void drawTrajectory(Canvas canvas, int x, int y, int w, int h) {
        canvas.drawText("Trajetória XY", x + 8, y + 28, textPaint);
        int margin = 40;
        // Draw axes
        canvas.drawLine(x + margin, y + h - margin, x + w - margin, y + h - margin, axisPaint);
        canvas.drawLine(x + margin, y + margin, x + margin, y + h - margin, axisPaint);

        drawPath(canvas, trajectoryWithZupt, x + margin, y + margin,
                w - 2*margin, h - 2*margin, zuPtPaint, true);
        drawPath(canvas, trajectoryNoZupt, x + margin, y + margin,
                w - 2*margin, h - 2*margin, noZuPtPaint, false);

        canvas.drawText("COM ZUPT", x + 8, y + h - 40, zuPtPaint);
        canvas.drawText("SEM ZUPT", x + 8, y + h - 18, noZuPtPaint);
    }

    private void drawPath(Canvas canvas, List<Pose> poses,
                          int ox, int oy, int pw, int ph, Paint paint, boolean isFirst) {
        if (poses.size() < 2) return;
        double minX = Double.MAX_VALUE, maxX = -Double.MAX_VALUE;
        double minY = Double.MAX_VALUE, maxY = -Double.MAX_VALUE;
        for (Pose p : poses) {
            minX = Math.min(minX, p.p[0]); maxX = Math.max(maxX, p.p[0]);
            minY = Math.min(minY, p.p[1]); maxY = Math.max(maxY, p.p[1]);
        }
        double rangeX = Math.max(maxX - minX, 0.01);
        double rangeY = Math.max(maxY - minY, 0.01);

        for (int i = 1; i < poses.size(); i++) {
            float x1 = ox + (float)((poses.get(i-1).p[0] - minX) / rangeX * pw);
            float y1 = oy + ph - (float)((poses.get(i-1).p[1] - minY) / rangeY * ph);
            float x2 = ox + (float)((poses.get(i).p[0] - minX) / rangeX * pw);
            float y2 = oy + ph - (float)((poses.get(i).p[1] - minY) / rangeY * ph);
            canvas.drawLine(x1, y1, x2, y2, paint);
        }
    }

    private void drawErrorCurves(Canvas canvas, int x, int y, int w, int h) {
        canvas.drawText("Erro de posição vs Tempo", x + 8, y + 28, textPaint);
        int margin = 40;
        canvas.drawLine(x + margin, y + h - margin, x + w - margin, y + h - margin, axisPaint);
        canvas.drawLine(x + margin, y + margin, x + margin, y + h - margin, axisPaint);
        canvas.drawText("t(s)", x + w - margin, y + h - margin + 18, textPaint);
        canvas.drawText("err(m)", x, y + margin, textPaint);

        drawCurve(canvas, errorsNoZupt, x + margin, y + margin, w - 2*margin, h - 2*margin, noZuPtPaint);
        drawCurve(canvas, errorsWithZupt, x + margin, y + margin, w - 2*margin, h - 2*margin, zuPtPaint);
    }

    private void drawCurve(Canvas canvas, List<Float> errors, int ox, int oy, int pw, int ph, Paint paint) {
        if (errors.size() < 2) return;
        float maxErr = 0;
        for (float e : errors) maxErr = Math.max(maxErr, e);
        if (maxErr < 0.001f) maxErr = 1f;

        for (int i = 1; i < errors.size(); i++) {
            float x1 = ox + (float)(i-1) / (errors.size()-1) * pw;
            float y1 = oy + ph - (errors.get(i-1) / maxErr) * ph;
            float x2 = ox + (float) i / (errors.size()-1) * pw;
            float y2 = oy + ph - (errors.get(i) / maxErr) * ph;
            canvas.drawLine(x1, y1, x2, y2, paint);
        }
    }
}
