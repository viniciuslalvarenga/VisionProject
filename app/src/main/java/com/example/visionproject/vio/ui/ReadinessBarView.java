package com.example.visionproject.vio.ui;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.View;

public class ReadinessBarView extends View {
    private double baselineRatio = 0;
    private double parallaxRatio = 0;

    private final Paint bgPaint   = new Paint();
    private final Paint barPaint  = new Paint();
    private final Paint textPaint = new Paint();

    public ReadinessBarView(Context ctx) { super(ctx); init(); }
    public ReadinessBarView(Context ctx, AttributeSet attrs) { super(ctx, attrs); init(); }
    public ReadinessBarView(Context ctx, AttributeSet attrs, int defStyle) {
        super(ctx, attrs, defStyle); init();
    }

    private void init() {
        bgPaint.setColor(Color.parseColor("#44000000"));
        bgPaint.setStyle(Paint.Style.FILL);
        textPaint.setColor(Color.WHITE);
        textPaint.setTextSize(28f);
        textPaint.setAntiAlias(true);
    }

    public void setReadiness(double baselineRatio, double parallaxRatio) {
        this.baselineRatio = baselineRatio;
        this.parallaxRatio = parallaxRatio;
        invalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        int w = getWidth(), h = getHeight();
        int barH = h / 2 - 8;
        int margin = 4;

        drawBar(canvas, "Baseline", baselineRatio, margin, margin, w - margin * 2, barH);
        drawBar(canvas, "Parallax", parallaxRatio, margin, h / 2 + margin, w - margin * 2, barH);
    }

    private void drawBar(Canvas canvas, String label, double ratio, int x, int y, int w, int h) {
        // Background
        RectF bg = new RectF(x, y, x + w, y + h);
        canvas.drawRect(bg, bgPaint);

        // Bar fill
        float fillW = (float) Math.min(ratio, 1.0) * w;
        barPaint.setColor(ratio >= 1.0 ? Color.GREEN : ratio >= 0.5 ? Color.parseColor("#FFA500") : Color.RED);
        canvas.drawRect(x, y, x + fillW, y + h, barPaint);

        // Label
        String text = label + ": " + (int)(ratio * 100) + "%";
        canvas.drawText(text, x + 8, y + h - 8, textPaint);
    }
}
