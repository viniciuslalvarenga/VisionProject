# CHECKLIST TÉCNICO - Módulo 3 (ESPEC_Calibracao) v3

## 1. Gestão de Memória e Recursos [OK]
- [x] **Mat.release()**: Todos os objetos `Mat` temporários no `ZhangCalibrationPipeline` e `CalibrationViewModel` são liberados explicitamente.
- [x] **Buffer da Câmera**: `lastFrame.release()` implementado no `onDestroy` da `CalibrationActivity`.
- [x] **Bitmap Recycling**: Prevenção de leaks em diálogos de comparação.

## 2. Performance de UI [OK]
- [x] **Zero Allocation em onDraw**: `ChessboardOverlayView` e `CoverageHeatmapView` usam buffers pré-alocados para desenhar pontos e regiões, evitando GC Thrashing.
- [x] **Taxa de Atualização**: Otimização do processamento de frames para manter >24 FPS durante a detecção.

## 3. Rastreabilidade Total (Total Traceability) [OK]
- [x] **Logging CSV**: Eventos de `FRAME_CAPTURED`, `FRAME_REJECTED` (com motivo), `CALIBRATION_DONE` e `PER_IMAGE_ERROR` implementados.
- [x] **Salvamento Unificado**: Ambos `.csv` e `.json` salvos em `Documents/VisionProject/` via `MediaStore` (Android 10+).
- [x] **Metadados**: Inclusão de modelo do dispositivo, versão do Android e timestamps ISO8601 nos logs.

## 4. Estabilidade do Algoritmo [OK]
- [x] **Filtro de Nitidez**: Bloqueio de capturas com `blurScore < 100.0`.
- [x] **Estabilidade de Pose**: Verificação de desvio padrão dos cantos antes da captura.
- [x] **Cooldown de Captura**: Intervalo de 800ms entre frames no modo Auto para evitar redundância.

## 5. Fluxo do Usuário [OK]
- [x] **Coleta Contínua**: Contador não trava em 15, permitindo melhorar o RMS progressivamente.
- [x] **Salvamento Inteligente**: Botão "Salvar" dispara calibração automática se os dados estiverem prontos mas não processados.
