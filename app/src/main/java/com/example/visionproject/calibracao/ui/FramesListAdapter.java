package com.example.visionproject.calibracao.ui;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageButton;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import com.example.visionproject.R;
import com.example.visionproject.calibracao.model.CalibrationFrame;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;

public class FramesListAdapter extends RecyclerView.Adapter<FramesListAdapter.ViewHolder> {

    public interface OnDeleteClickListener {
        void onDelete(int originalIndex);
    }

    private final List<FrameItem> items = new ArrayList<>();
    private final OnDeleteClickListener onDelete;
    private boolean showErrors = false;

    private static class FrameItem {
        final CalibrationFrame frame;
        final int originalIndex;
        Double error = null;

        FrameItem(CalibrationFrame frame, int originalIndex) {
            this.frame = frame;
            this.originalIndex = originalIndex;
        }
    }

    public FramesListAdapter(OnDeleteClickListener onDelete) {
        this.onDelete = onDelete;
    }

    public void setData(List<CalibrationFrame> frames, List<Double> errors) {
        items.clear();
        showErrors = errors != null && !errors.isEmpty() && errors.size() == frames.size();
        
        for (int i = 0; i < frames.size(); i++) {
            FrameItem item = new FrameItem(frames.get(i), i);
            if (showErrors) {
                item.error = errors.get(i);
            }
            items.add(item);
        }

        if (showErrors) {
            // Ordenar por erro decrescente
            Collections.sort(items, (a, b) -> Double.compare(b.error, a.error));
        }
        notifyDataSetChanged();
    }

    @NonNull
    @Override
    public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext()).inflate(R.layout.cal_item_frame, parent, false);
        return new ViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull ViewHolder holder, int position) {
        FrameItem item = items.get(position);
        holder.tvIdx.setText(String.valueOf(item.originalIndex));
        holder.tvBlur.setText(String.format(Locale.US, "%.1f", item.frame.getBlurScore()));
        holder.tvStability.setText(String.format(Locale.US, "%.2f", item.frame.getPoseStabilityScore()));
        
        if (showErrors && item.error != null) {
            holder.tvError.setVisibility(View.VISIBLE);
            holder.tvError.setText(String.format(Locale.US, "%.3f", item.error));
        } else {
            holder.tvError.setVisibility(View.GONE);
        }

        holder.btnDelete.setOnClickListener(v -> {
            if (onDelete != null) {
                onDelete.onDelete(item.originalIndex);
            }
        });
    }

    @Override
    public int getItemCount() {
        return items.size();
    }

    static class ViewHolder extends RecyclerView.ViewHolder {
        TextView tvIdx, tvBlur, tvStability, tvError;
        ImageButton btnDelete;

        ViewHolder(View itemView) {
            super(itemView);
            tvIdx = itemView.findViewById(R.id.cal_item_idx);
            tvBlur = itemView.findViewById(R.id.cal_item_blur);
            tvStability = itemView.findViewById(R.id.cal_item_stability);
            tvError = itemView.findViewById(R.id.cal_item_error);
            btnDelete = itemView.findViewById(R.id.cal_item_btn_delete);
        }
    }
}
