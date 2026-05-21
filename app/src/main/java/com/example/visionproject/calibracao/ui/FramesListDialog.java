package com.example.visionproject.calibracao.ui;

import android.app.Dialog;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.DialogFragment;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.example.visionproject.R;
import com.example.visionproject.calibracao.CalibrationViewModel;
import com.example.visionproject.calibracao.model.CalibrationResult;
import com.example.visionproject.calibracao.repository.CalibrationFramesRepository;

public class FramesListDialog extends DialogFragment {

    private final CalibrationViewModel viewModel;

    public FramesListDialog(CalibrationViewModel viewModel) {
        this.viewModel = viewModel;
    }

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        View view = inflater.inflate(R.layout.cal_dialog_frames_list, container, false);

        RecyclerView rv = view.findViewById(R.id.cal_rv_frames);
        rv.setLayoutManager(new LinearLayoutManager(getContext()));

        FramesListAdapter adapter = new FramesListAdapter(index -> {
            viewModel.deleteFrame(index);
            // Refresh adapter data
            updateData(rv, (FramesListAdapter) rv.getAdapter());
        });
        rv.setAdapter(adapter);

        View headerError = view.findViewById(R.id.cal_header_error);
        
        viewModel.getResult().observe(getViewLifecycleOwner(), result -> {
            headerError.setVisibility(result != null ? View.VISIBLE : View.GONE);
            updateData(rv, adapter);
        });

        view.findViewById(R.id.cal_btn_close_dialog).setOnClickListener(v -> dismiss());

        updateData(rv, adapter);

        return view;
    }

    private void updateData(RecyclerView rv, FramesListAdapter adapter) {
        CalibrationResult res = viewModel.getResult().getValue();
        adapter.setData(
                CalibrationFramesRepository.getInstance().getAll(),
                res != null ? res.getPerImageReprojectionError() : null
        );
    }

    @Override
    public void onStart() {
        super.onStart();
        Dialog dialog = getDialog();
        if (dialog != null) {
            int width = ViewGroup.LayoutParams.MATCH_PARENT;
            int height = ViewGroup.LayoutParams.MATCH_PARENT;
            dialog.getWindow().setLayout(width, height);
        }
    }
}
