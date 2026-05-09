package com.example.visionproject.modelocamera;

import androidx.annotation.NonNull;
import androidx.lifecycle.ViewModel;
import androidx.lifecycle.ViewModelProvider;

import com.example.visionproject.modelocamera.repository.CalibrationRepository;
import com.example.visionproject.modelocamera.strategy.ImageUndistortStrategy;
import com.example.visionproject.modelocamera.strategy.PointUndistortStrategy;

/**
 * Factory para criar instâncias de ModeloCameraViewModel com injeção de dependências manual.
 */
public class ModeloCameraVMFactory implements ViewModelProvider.Factory {

    private final android.app.Application application;

    public ModeloCameraVMFactory(android.app.Application application) {
        this.application = application;
    }

    @NonNull
    @Override
    @SuppressWarnings("unchecked")
    public <T extends ViewModel> T create(@NonNull Class<T> modelClass) {
        if (modelClass.isAssignableFrom(ModeloCameraViewModel.class)) {
            return (T) new ModeloCameraViewModel(
                    application,
                    new ImageUndistortStrategy(),
                    new PointUndistortStrategy(),
                    CalibrationRepository.getInstance()
            );
        }
        throw new IllegalArgumentException("Classe ViewModel desconhecida");
    }
}
