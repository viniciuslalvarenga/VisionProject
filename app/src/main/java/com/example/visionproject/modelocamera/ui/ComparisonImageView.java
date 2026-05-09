package com.example.visionproject.modelocamera.ui;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.AttributeSet;
import android.view.LayoutInflater;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;

import androidx.annotation.Nullable;

import com.example.visionproject.R;

/**
 * Custom View que exibe a comparação entre a imagem original e a corrigida.
 */
public class ComparisonImageView extends LinearLayout {

    private ImageView ivOriginal;
    private ImageView ivCorrected;
    private TextView tvLabelOriginal;
    private TextView tvLabelCorrected;

    public ComparisonImageView(Context context) {
        super(context);
        init(context);
    }

    public ComparisonImageView(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
        init(context);
    }

    private OnPointSelectedListener listener;

    public interface OnPointSelectedListener {
        void onPointSelected(org.opencv.core.Point p);
    }

    public void setOnPointSelectedListener(OnPointSelectedListener listener) {
        this.listener = listener;
    }

    private void init(Context context) {
        setOrientation(HORIZONTAL);
        setWeightSum(2);
        LayoutInflater.from(context).inflate(R.layout.mc_comparison_view, this, true);

        ivOriginal = findViewById(R.id.mc_iv_original);
        ivCorrected = findViewById(R.id.mc_iv_corrected);
        tvLabelOriginal = findViewById(R.id.mc_tv_label_original);
        tvLabelCorrected = findViewById(R.id.mc_tv_label_corrected);

        ivOriginal.setOnTouchListener((v, event) -> {
            if (event.getAction() == android.view.MotionEvent.ACTION_DOWN) {
                v.performClick();
                if (listener != null) {
                    float x = event.getX();
                    float y = event.getY();
                    // Conversão simplificada (assume fitCenter)
                    android.graphics.drawable.Drawable drawable = ivOriginal.getDrawable();
                    if (drawable != null) {
                        int intrinsicWidth = drawable.getIntrinsicWidth();
                        int intrinsicHeight = drawable.getIntrinsicHeight();
                        float[] values = new float[9];
                        ivOriginal.getImageMatrix().getValues(values);
                        float transX = values[android.graphics.Matrix.MTRANS_X];
                        float transY = values[android.graphics.Matrix.MTRANS_Y];
                        float scaleX = values[android.graphics.Matrix.MSCALE_X];
                        float scaleY = values[android.graphics.Matrix.MSCALE_Y];

                        float imgX = (x - transX) / scaleX;
                        float imgY = (y - transY) / scaleY;

                        if (imgX >= 0 && imgX <= intrinsicWidth && imgY >= 0 && imgY <= intrinsicHeight) {
                            listener.onPointSelected(new org.opencv.core.Point(imgX, imgY));
                        }
                    }
                }
                return true;
            }
            return false;
        });

        ivOriginal.setOnClickListener(v -> {
            // Placeholder for accessibility
        });
    }

    public void setOriginal(Bitmap bitmap) {
        if (ivOriginal != null) ivOriginal.setImageBitmap(bitmap);
    }

    public void setCorrected(Bitmap bitmap) {
        if (ivCorrected != null) ivCorrected.setImageBitmap(bitmap);
    }
}
