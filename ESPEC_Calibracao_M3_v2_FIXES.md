# ESPECIFICAÇÃO TÉCNICA v2 — Correções pós-auditoria do Módulo 3
**Projeto:** VisionProject (Android, Java + OpenCV4Android)
**Atividade:** Módulo 3 — Calibração de Câmera (Método de Zhang)
**Pré-requisito:** ESPEC_Calibracao_M3.md (v1) já implementada — **TODAS as classes existentes continuam funcionando**

---

## 0. Princípios Não-Negociáveis (REVISÃO)

1. **Zero alteração no PccModule** (Módulo 1) — continua valendo.
2. **Zero alteração na API pública** das classes do Módulo 2 (`modelocamera/`) — pode estender, não remover.
3. **Zero alteração na API pública** das classes do Módulo 3 que estão em uso — pode adicionar métodos, não mudar assinatura existente.
4. **Esta v2 é REFATORAÇÃO + UM novo botão.** Funcionalidade nova mínima. O foco é corrigir desvios apontados na auditoria sem quebrar o que já está rodando.
5. **Smoke test obrigatório** após implementar: PCC funciona → ModeloCamera funciona → Calibração funciona → CSV é gerado → JSON é exportado → ModeloCamera carrega valores calibrados.

---

## 1. Correções Ordenadas por Prioridade

### 🔴 PRIORIDADE 1 — Estabilidade da auto-captura (BUG #8 da auditoria)

**Problema:** `PoseStabilityStrategy.checkStability()` verifica apenas o último shift entre dois frames consecutivos, e não a estabilidade ao longo da janela inteira de N frames. Resultado: auto-captura pode disparar prematuramente em movimento contínuo lento, capturando frames borrados ou em transição.

**Arquivo:** `app/src/main/java/com/example/visionproject/calibracao/strategy/PoseStabilityStrategy.java`

**O que fazer:**
1. Substituir o método `checkStability(MatOfPoint2f currentCorners)` para que:
   - Adicione o frame atual ao histórico (max N=5).
   - Só retorne `true` quando **todos os shifts da janela** (entre frames consecutivos da fila) forem menores que `threshold`.
   - Calcule o **shift máximo** da janela (não a média) — critério mais conservador.
2. **Remover** o método `isStable()` (mantido só por compatibilidade — ninguém chama).
3. **Remover** o método `calculateTotalHistoryStability()` (dead code, retorna 0).
4. Manter `reset()`, `calculateMeanShift()` e o construtor com configuração de tamanho/threshold.

**Implementação sugerida do novo `checkStability`:**
```java
public boolean checkStability(MatOfPoint2f currentCorners) {
    if (currentCorners == null) {
        reset();
        return false;
    }

    // clone defensivo
    MatOfPoint2f clone = new MatOfPoint2f();
    currentCorners.copyTo(clone);

    // Mantém janela de tamanho fixo
    if (history.size() >= historySize) {
        MatOfPoint2f removed = history.poll();
        if (removed != null) removed.release();
    }
    history.add(clone);

    // Precisa pelo menos N frames para julgar estabilidade
    if (history.size() < historySize) return false;

    // Verifica shift MÁXIMO entre pares consecutivos da janela
    MatOfPoint2f[] arr = history.toArray(new MatOfPoint2f[0]);
    double maxShift = 0;
    for (int i = 1; i < arr.length; i++) {
        double s = calculateMeanShift(arr[i-1], arr[i]);
        if (s > maxShift) maxShift = s;
    }
    return maxShift < threshold;
}
```

**Critério de aceite:**
- Mover o smartphone lentamente sobre o tabuleiro NÃO deve disparar captura automática (precisa parar por ~1 segundo).
- Manter o smartphone parado por ~1 segundo deve disparar exatamente uma captura.

---

### 🔴 PRIORIDADE 2 — Logger CSV em background thread (ANTI-PADRÃO #2 da spec original)

**Problema:** `CalibrationCsvLogger.saveSession()` escreve no thread chamador (UI). Pode travar a interface por algumas centenas de ms ao salvar a sessão, especialmente se o CSV crescer.

**Arquivo:** `app/src/main/java/com/example/visionproject/calibracao/repository/CalibrationCsvLogger.java`

**O que fazer:**
1. Adicionar campo `private final ExecutorService io = Executors.newSingleThreadExecutor();` no Singleton.
2. Em `saveSession(Context context)`:
   - Capturar `data` e `fileName` no thread atual (rápido).
   - Submeter a escrita do arquivo via `io.execute(() -> { /* MediaStore write */ });`.
   - **NÃO** chamar `io.shutdown()` — Singleton vive até o app morrer.
3. Em `logFrameCaptured()`, `logCalibrationDone()`, `logEvent()`: também submeter o `appendLine(...)` ao executor (escritas no buffer também ficam fora do thread chamador).
4. **Sincronizar** o `logBuffer` (use `synchronized (logBuffer)` ou `StringBuffer`) — múltiplos threads podem escrever simultaneamente.

**Critério de aceite:**
- Pressionar "Salvar" não congela a UI por mais de 50ms (perceptível visualmente).
- O CSV ainda é gerado corretamente em `Documents/VisionProject/`.

---

### 🟡 PRIORIDADE 3 — Eventos faltantes no CSV

**Problema:** O logger só registra 4 tipos de evento. Faltam 6 que ajudam muito na análise temporal do experimento e na narrativa do relatório.

**Arquivo:** `app/src/main/java/com/example/visionproject/calibracao/repository/CalibrationCsvLogger.java` + chamadores no `CalibrationViewModel`

**O que adicionar:**

| Evento | Onde disparar | Campos importantes |
|---|---|---|
| `SESSION_START` | No construtor do Singleton (uma vez na inicialização) | timestamp_iso, device_model, android_version, notes="App started" |
| `FRAME_DETECTED` | `CalibrationViewModel.onPreviewFrame` quando `detection.found == true` (rate-limited a 1 evento/segundo para não inflar CSV) | corners_found, blur_score |
| `FRAME_REJECTED` | Quando frame seria capturado mas não passa nos critérios (blur, cantos incompletos, manual delete) | rejection_reason ("blur"\|"incomplete_corners"\|"manual_delete"), blur_score |
| `CALIBRATION_STARTED` | Início de `runCalibration()` no ViewModel | images_used, notes (ex: "Starting calibration with 22 frames") |
| `JSON_EXPORTED` | Após `CalibrationJsonStore.save()` em `saveResult()` | json_path |
| `UNDISTORT_PREVIEW` | Toda vez que `generateUndistortPreview()` é chamado | image_w, image_h, elapsed_ms |

**Como implementar:**
1. Adicionar métodos públicos no logger: `logFrameDetected(int corners, double blur)`, `logFrameRejected(String reason, double blur)`, `logCalibrationStarted(int images)`, `logJsonExported(String path)`, `logUndistortPreview(int w, int h, long elapsed)`.
2. Cada método monta uma linha CSV com APENAS os campos do seu evento preenchidos (resto vazio).
3. Chamar do `CalibrationViewModel`:
   - `onPreviewFrame`: rate-limit com `lastDetectedLogMs + 1000 < now` e logar.
   - Capture rejeitado: chamar `logFrameRejected(...)` em vez de só ignorar silenciosamente.
   - `runCalibration`: `logger.logCalibrationStarted(framesRepo.size())` antes do `executor.execute`.
   - `saveResult`: após `CalibrationJsonStore.save(...)`, chamar `logger.logJsonExported(jsonPath)` (precisa retornar o path do `save`).
   - `generateUndistortPreview`: registrar antes do `postValue`.

**Modificar `CalibrationJsonStore.save()`** para retornar `String` com o caminho gerado:
```java
public static String save(CalibrationResult result, Context context) {
    // ... salva interno e externo
    return externalFile.getAbsolutePath();
}
```

**Critério de aceite:** Após uma sessão completa (capturar 20 frames + calibrar + salvar + 1 undistort preview), o CSV deve conter pelo menos 1 linha de cada um dos 10 event types.

---

### 🟡 PRIORIDADE 4 — `pose_stability_score` real (não placeholder)

**Problema:** No `CalibrationViewModel.captureFrame()`, o `stability` é hardcoded como `0.0`. Esse campo no CSV sempre aparece como zero.

**Arquivo:** `app/src/main/java/com/example/visionproject/calibracao/CalibrationViewModel.java`

**O que fazer:**
1. Adicionar método público em `PoseStabilityStrategy`:
   ```java
   public double getCurrentStabilityScore() {
       if (history.size() < 2) return Double.NaN;
       MatOfPoint2f[] arr = history.toArray(new MatOfPoint2f[0]);
       double maxShift = 0;
       for (int i = 1; i < arr.length; i++) {
           double s = calculateMeanShift(arr[i-1], arr[i]);
           if (s > maxShift) maxShift = s;
       }
       return maxShift;
   }
   ```
2. No `CalibrationViewModel`, quando capturar:
   ```java
   double stability = stabilityStrategy.getCurrentStabilityScore();
   captureFrame(detection.corners, blurScore, stability);
   ```

**Critério de aceite:** No CSV, a coluna `pose_stability_score` deve ter valores reais (entre 0.0 e ~5.0 px), não zero.

---

### 🟡 PRIORIDADE 5 — Reaproveitar `CsvWriter` e `CsvFormatter` (ANTI-PADRÃO #1)

**Problema:** O logger duplicou a lógica de escrita CSV em StringBuilder próprio em vez de usar `shared/csv/CsvWriter.java` e `CsvFormatter.java` (já existentes do Módulo 2).

**Arquivos:** `CalibrationCsvLogger.java` e (já existem) `shared/csv/CsvWriter.java`, `shared/csv/CsvFormatter.java`

**O que fazer:**
1. Refatorar `CalibrationCsvLogger` para usar:
   - `CsvFormatter.fmtDouble(...)`, `CsvFormatter.fmtTimestampIso(...)`, `CsvFormatter.escape(...)` em vez de `String.format` direto.
   - `CsvWriter` com `writeHeader(String[])` e `writeRow(Map<String,String>)`.
2. **Não mudar** o nome do arquivo gerado nem a ordem das colunas (compatibilidade com scripts Python que rodam sobre o CSV).
3. Manter o Holder Singleton.

**Crítério de aceite:** Comportamento idêntico do CSV anterior (mesmo header, mesmo nome de arquivo, mesma ordem de colunas), mas com 70% menos código no logger.

> **Se essa refatoração ficar arriscada para o prazo, pode pular** — é melhor enviar com o logger duplicado e CSV correto do que quebrar a coleta de dados. Marcar como "trabalho futuro".

---

### 🟢 PRIORIDADE 6 — Implementar `getDistortionType()` em `CalibrationResult`

**Problema:** `CalibrationResult` expõe `getK1()` etc. mas não classifica BARREL/PINCUSHION/NONE como o `DistortionCoefficients` do Módulo 2.

**Arquivo:** `app/src/main/java/com/example/visionproject/calibracao/model/CalibrationResult.java`

**O que fazer:**
1. Adicionar método:
   ```java
   public DistortionCoefficients.DistortionType getDistortionType() {
       double k1 = getK1();
       if (k1 < -0.05) return DistortionCoefficients.DistortionType.BARREL;
       if (k1 > 0.05) return DistortionCoefficients.DistortionType.PINCUSHION;
       return DistortionCoefficients.DistortionType.NONE;
   }
   ```
2. Adicionar coluna `distortion_type` ao CSV em `CALIBRATION_DONE`.
3. Adicionar campo no JSON exportado:
   ```json
   "distortion": { "k1": ..., "k2": ..., ..., "type": "BARREL" }
   ```

**Critério de aceite:** Após uma calibração de smartphone com lente principal, o tipo deve ser `BARREL` (k1 negativo). Se aparecer `PINCUSHION` ou `NONE`, alguma coisa está errada na coleta.

---

## 2. Novo Botão Funcional: "Lista de Frames" (com erro por imagem)

### 🆕 Por que precisa

Para atingir RMS < 1.0 px, o usuário precisa identificar e remover os frames que mais contribuem para o erro. Atualmente:
- O CSV registra `PER_IMAGE_ERROR` por frame após calibração ✅
- Mas o app **não mostra esses erros na tela**, e **não permite deletar frames individuais**.
- Resultado: o usuário precisa abrir o CSV no PC, identificar outliers, voltar ao app, "Limpar" tudo, recapturar 20 frames. **Inviável na prática.**

### O que implementar

**Adicionar botão `[📋 Lista]` na barra de controles do layout `cal_activity_calibracao.xml`**, posicionado entre `[Limpar]` e `[Calibrar]`.

**Comportamento do clique:**
- Antes da calibração (`state != DONE`): abre dialog mostrando os frames coletados, com:
  - Índice
  - Timestamp
  - Blur score
  - Pose stability score
  - Botão `[🗑]` por linha para deletar individualmente
- Após a calibração (`state == DONE`): mesma dialog mas com **coluna adicional "Erro reprojeção (px)"** ordenando os frames do maior erro para o menor. Permite identificar e deletar os 2-3 piores e re-rodar `runCalibration()` sem perder os bons.

### Onde implementar

1. **Novo layout** `app/src/main/res/layout/cal_dialog_frames_list.xml`:
   - `RecyclerView` com `LinearLayoutManager` vertical.
   - Botão `[Fechar]` no rodapé.

2. **Novo layout** `app/src/main/res/layout/cal_item_frame.xml`:
   - `LinearLayout` horizontal com:
     - `TextView` índice
     - `TextView` blur
     - `TextView` stability
     - `TextView` erro_reprojection (visível apenas após calibração)
     - `ImageButton` ❌ deletar

3. **Nova classe `ui/FramesListAdapter.java`** (RecyclerView.Adapter):
   - Recebe `List<CalibrationFrame>` + opcional `List<Double> perImageErrors`.
   - Callback `OnDeleteClickListener onDelete`.

4. **Nova classe `ui/FramesListDialog.java`** (DialogFragment ou simples Dialog):
   - Recebe `CalibrationViewModel viewModel`.
   - Observa `result` LiveData para mostrar/esconder coluna de erro.
   - No clique de deletar: chama `viewModel.deleteFrame(index)`.

5. **Adicionar método ao `CalibrationViewModel`:**
   ```java
   public void deleteFrame(int index) {
       framesRepo.removeAt(index);
       framesCollectedCount.postValue(framesRepo.size());
       // Logar evento
       CalibrationCsvLogger.getInstance().logFrameRejected("manual_delete", -1);
       // Se já calibrou, recalcular automaticamente ou mudar estado
       if (state.getValue() == CalibrationState.DONE) {
           state.postValue(CalibrationState.READY_TO_CALIBRATE);
           result.postValue(null);  // forca usuario a re-calibrar
       }
   }
   ```

6. **Adicionar string** em `app/src/main/res/values/strings.xml`:
   ```xml
   <string name="cal_btn_list">📋 Lista</string>
   <string name="cal_dialog_title_list">Frames coletados</string>
   <string name="cal_col_error">Erro (px)</string>
   ```

7. **No `CalibrationActivity`**, adicionar listener:
   ```java
   findViewById(R.id.cal_btn_list).setOnClickListener(v -> {
       new FramesListDialog(viewModel).show(getSupportFragmentManager(), "frames_list");
   });
   ```

### Critério de aceite
- Botão "Lista" aparece nos controles.
- Antes de calibrar: mostra N frames com blur/stability scores.
- Depois de calibrar: mostra N frames + coluna de erro de reprojeção, ordenados pelo maior erro.
- Deletar 1 frame remove ele do conjunto e atualiza contador.
- Re-calibrar com o conjunto reduzido funciona.
- Cada deleção manual gera um `FRAME_REJECTED` no CSV com `reason="manual_delete"`.

---

## 3. Critérios de Aceite (DoD) consolidados

- [ ] PCC continua funcionando (smoke test).
- [ ] ModeloCamera continua funcionando (smoke test).
- [ ] Auto-captura **só dispara após pose estável por ~1 segundo** (não em movimento lento).
- [ ] Salvar CSV não trava UI.
- [ ] CSV contém os 10 event types listados (ou pelo menos o subset implementado se P5 foi pulado).
- [ ] Coluna `pose_stability_score` no CSV tem valores reais (não zero).
- [ ] CSV e JSON contêm campo `distortion_type` com valor `BARREL`/`PINCUSHION`/`NONE`.
- [ ] **Novo botão "Lista" funciona** e permite deletar frames.
- [ ] Após deletar 2-3 frames com maior erro e re-calibrar, o RMS diminui.
- [ ] Testes unitários existentes ainda passam.

---

## 4. Anti-padrões a evitar (lembretes)

- ❌ **Não** rodar `calibrateCamera()` no main thread (já está OK).
- ❌ **Não** criar segundo Singleton para o logger (Holder atual está OK, só refatorar para usar Executor).
- ❌ **Não** quebrar contrato público do `CalibrationViewModel` — só adicionar métodos.
- ❌ **Não** mudar o nome do arquivo CSV ou JSON (scripts Python esperam `cal_session_*.csv` e `calibration.json`).
- ❌ **Não** mudar a ordem das colunas do CSV.
- ❌ **Não** alterar `PccModule.java` nem `ModeloCameraActivity` (exceto se a P3 do logger exigir alterar `ModeloCameraActivity.onResume` — improvável).
- ❌ **Não** introduzir libs novas (sem Glide, Picasso, Room, Hilt, etc.). Use o que já está no projeto.

---

## 5. Resumo executivo (cheat-sheet)

1. Refatorar `PoseStabilityStrategy.checkStability()` para verificar shift máximo da janela inteira.
2. Migrar escrita do `CalibrationCsvLogger` para `ExecutorService` single-thread.
3. Adicionar 5-6 novos métodos `log*()` no logger e chamá-los do ViewModel.
4. Calcular `pose_stability_score` real e passar para `captureFrame()`.
5. (Opcional) Refatorar logger para usar `CsvWriter`/`CsvFormatter`.
6. Adicionar `getDistortionType()` em `CalibrationResult` + campo no CSV/JSON.
7. **Implementar botão "📋 Lista"** com dialog + RecyclerView + deleção individual.
8. Testar smoke: PCC ok, ModeloCamera ok, calibração ok, lista ok, exportação ok.
9. Validar critérios da seção 3.

---

**Autor:** Vinicius L. Alvarenga
**Versão:** 2.0 (correções pós-auditoria de v1)
**Data:** Maio/2026
**Pré-requisitos:** ESPEC_Calibracao_M3.md (v1) já implementada e funcional.
