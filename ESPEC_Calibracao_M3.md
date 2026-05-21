# ESPECIFICAÇÃO TÉCNICA — Módulo 3: Calibração de Câmera (Método de Zhang)
**Projeto:** VisionProject (Android, Java + OpenCV4Android)
**Atividade:** Módulo 3 — Aula Teórica 4 + Prática 4 — **M1 (Primeira Entrega Avaliada)**
**Pré-requisitos:** Módulo 1 (PCC) e Módulo 2 (ModeloCamera) já implementados — **NENHUM PODE SER REGREDIDO**

---

## 0. Princípios Não-Negociáveis (REVISÃO CRÍTICA)

Antes de escrever uma linha de código, **leia toda esta seção e marque mentalmente cada item**.

1. **Zero alteração no PccModule** (Módulo 1) — continua valendo da v1 e v2.
2. **Zero alteração nas classes existentes** do `modelocamera/` que já funcionam — pode **estender** o `CalibrationRepository` para receber valores reais calibrados, mas **não mudar** sua API pública nem remover métodos.
3. **Tudo em pacote novo:** `com.example.visionproject.calibracao`.
4. **MVVM obrigatório.** Activity faz binding, ViewModel orquestra, classes de domínio guardam estado.
5. **Reaproveitar OpenCV** já inicializado pelos módulos anteriores. Não criar segunda inicialização.
6. **Reaproveitar logger CSV** do Módulo 2 (`shared/csv/CsvWriter`, `shared/csv/CsvFormatter`). Não duplicar.
7. **Permissões CAMERA e STORAGE** já estão concedidas pelos módulos anteriores. Não duplicar pedidos.
8. **Navegação:** adicionar **um novo botão** "Calibração" na MainActivity, sem remover os existentes.
9. **Calibração é entrega avaliada (M1).** Toda decisão de design deve priorizar (a) RMS de reprojeção < 1.0 px, (b) rastreabilidade total via CSV, (c) reuso pelos módulos seguintes via `calibration.json`.

---

## 1. Estrutura de Pacotes

```
com.example.visionproject/
├── pcc/                                  [JÁ EXISTE — INTACTO]
├── modelocamera/                         [JÁ EXISTE — INTACTO; pode ser CONSUMIDOR]
│   └── repository/CalibrationRepository  [JÁ EXISTE — pode ganhar método loadFromJson()]
├── calibracao/                           [NOVO]
│   ├── CalibrationActivity.java          (UI principal)
│   ├── CalibrationViewModel.java         (orquestração)
│   ├── CalibrationState.java             (enum: IDLE, DETECTING, CAPTURING, READY_TO_CALIBRATE, CALIBRATING, DONE, ERROR)
│   ├── model/
│   │   ├── CalibrationFrame.java         (1 imagem capturada com seus cantos)
│   │   ├── CalibrationResult.java        (resultado final: K, D, RMS, per-image errors)
│   │   ├── PatternSpec.java              (interface)
│   │   └── ChessboardSpec.java           (9×6 quadrados, 25mm)
│   ├── strategy/                         (Strategy pattern)
│   │   ├── PatternDetectionStrategy.java (interface)
│   │   ├── ChessboardDetectionStrategy.java
│   │   ├── BlurDetectionStrategy.java    (variância de Laplaciano)
│   │   └── PoseStabilityStrategy.java    (cantos não se movem por N frames)
│   ├── factory/
│   │   └── ObjectPointsFactory.java      (gera grade 3D Z=0)
│   ├── pipeline/
│   │   ├── ZhangCalibrationPipeline.java (executa calibrateCamera + interpreta)
│   │   └── ReprojectionErrorAnalyzer.java
│   ├── repository/
│   │   ├── CalibrationFramesRepository.java (Singleton: lista de frames coletados)
│   │   └── CalibrationJsonStore.java     (serializa/desserializa calibration.json)
│   └── ui/
│       ├── ChessboardOverlayView.java    (camera preview + overlay verde dos cantos)
│       ├── CoverageHeatmapView.java      (mapa de calor das regiões cobertas)
│       └── BeforeAfterUndistortView.java (frame original | corrigido)
└── shared/                               [JÁ EXISTE]
    ├── csv/CsvWriter.java                [REUTILIZAR]
    ├── csv/CsvFormatter.java             [REUTILIZAR]
    └── DeviceInfoProvider.java           [REUTILIZAR]
```

---

## 2. Padrões de Design Aplicados

| Padrão | Onde | Por quê |
|---|---|---|
| **MVVM** | toda Activity | Separação UI / lógica / dados |
| **Strategy** | `PatternDetectionStrategy` (chessboard vs futuros padrões), `BlurDetectionStrategy` (Laplacian vs outros), `PoseStabilityStrategy` | Trocar critério sem mudar consumidor |
| **State** | `CalibrationState` enum + transições no ViewModel | UI reflete claramente o estágio do pipeline |
| **Singleton** | `CalibrationFramesRepository` (Holder pattern thread-safe) | Lista de frames sobrevive a rotação |
| **Factory** | `ObjectPointsFactory` | Gerar grade 3D para qualquer rows×cols×squareSize |
| **Builder** | `CalibrationConfig.Builder` | Configuração com 6+ parâmetros sem ambiguidade |
| **Observer** (LiveData) | ViewModel ↔ Activity | Reatividade |
| **Repository** | `CalibrationFramesRepository`, `CalibrationJsonStore` | Abstrair fonte de dados |
| **Pipeline / Template Method** | `ZhangCalibrationPipeline` | Etapas fixas (validate → calibrate → analyze → save) |

---

## 3. Lista Ordenada de Implementação

### 🔹 Passo 1 — Modelos de domínio (puros, sem Android)

#### 1.1 `model/PatternSpec.java` (interface)
```java
public interface PatternSpec {
    Size getInternalCornersSize();    // Size(cols, rows) — para chessboard 9x6 = (9,6)
    float getSquareSizeMm();
    Mat generateObjectPoints();       // Mat (rows*cols × 3) com Z=0 e XY em mm
    int getDetectionFlag();           // ex: Calib3d.CALIB_CB_ADAPTIVE_THRESH | NORMALIZE_IMAGE
}
```

#### 1.2 `model/ChessboardSpec.java`
- Implementa `PatternSpec`.
- Construtor: `ChessboardSpec(int cols, int rows, float squareSizeMm)`.
- Default: 9 cols, 6 rows, 25.0 mm.
- `generateObjectPoints()` gera `Mat` `CV_32FC3` com `(c*sq, r*sq, 0)` para cada par (r, c).
- Validação no construtor: `cols >= 3`, `rows >= 3`, `squareSizeMm > 0`.

#### 1.3 `model/CalibrationFrame.java`
- Imutável.
- Campos: `int index`, `long timestampMs`, `MatOfPoint2f corners` (ou `Mat`), `int cornersFound`, `double blurScore`, `double poseStabilityScore`, `Rect coverageBoundingBox`, `String thumbnailPath` (opcional).
- Sem setters; tudo via construtor.

#### 1.4 `model/CalibrationResult.java`
- Imutável.
- Campos:
  - `double rms` (global)
  - `Mat cameraMatrix` (K 3×3)
  - `Mat distCoeffs` (1×5: k1, k2, p1, p2, k3)
  - `List<Double> perImageReprojectionError` (RMS por imagem)
  - `int imagesUsed`, `int imagesRejected`
  - `long elapsedMsTotal`
  - `Date calibrationDateTime`
  - `Size imageSize`
- Métodos derivados:
  - `getFx(), getFy(), getCx(), getCy(), getSkew()`
  - `getK1(), getK2(), getP1(), getP2(), getK3()`
  - `getDistortionType()` (BARREL / PINCUSHION / NONE) — reutiliza enum do Módulo 2
  - `toJson(): JSONObject`

### 🔹 Passo 2 — Detecção e validação (Strategies)

#### 2.1 `strategy/PatternDetectionStrategy.java`
```java
public interface PatternDetectionStrategy {
    DetectionResult detect(Mat grayFrame, PatternSpec pattern);
}
public class DetectionResult {
    public final boolean found;
    public final MatOfPoint2f corners;
    public final long elapsedMs;
    // construtor
}
```

#### 2.2 `strategy/ChessboardDetectionStrategy.java`
- Implementa o algoritmo:
  1. `Calib3d.findChessboardCorners(gray, pattern.getInternalCornersSize(), corners, pattern.getDetectionFlag())`.
  2. Se `found`: refinar com `Imgproc.cornerSubPix(gray, corners, new Size(11,11), new Size(-1,-1), tc)` onde `tc = new TermCriteria(EPS+COUNT, 30, 0.001)`.
  3. Retornar `DetectionResult(found, corners, elapsedMs)`.

#### 2.3 `strategy/BlurDetectionStrategy.java`
- Implementa critério **variância do Laplaciano** (clássico em OpenCV):
  ```java
  Mat lap = new Mat();
  Imgproc.Laplacian(gray, lap, CvType.CV_64F);
  MatOfDouble mean = new MatOfDouble(), stdDev = new MatOfDouble();
  Core.meanStdDev(lap, mean, stdDev);
  double variance = Math.pow(stdDev.get(0,0)[0], 2);
  ```
- Threshold default: `variance >= 100.0` é considerada "nítida" (sem blur). Ajustar empiricamente.
- Método `boolean isSharp(Mat gray)` e `double getScore(Mat gray)`.

#### 2.4 `strategy/PoseStabilityStrategy.java`
- Mantém histórico das últimas N (default 5) detecções de cantos.
- Compara distância média Euclidiana entre cantos consecutivos.
- Se `meanShift < 2.0 px` por 5 frames consecutivos → pose estável → **dispara captura automática**.
- Estado interno: `Queue<MatOfPoint2f> history` (LinkedList).
- Método `boolean isStable(MatOfPoint2f currentCorners)`.
- Método `void reset()`.

### 🔹 Passo 3 — Factory e Repository

#### 3.1 `factory/ObjectPointsFactory.java`
```java
public class ObjectPointsFactory {
    public static Mat createPlanarGrid(int cols, int rows, float squareSizeMm) { /* CV_32FC3, Z=0 */ }
    public static Mat createForPattern(PatternSpec p) { return p.generateObjectPoints(); }
}
```

#### 3.2 `repository/CalibrationFramesRepository.java` (Singleton)
- Holder pattern thread-safe.
- Mantém `List<CalibrationFrame> frames` (sincronizada).
- Métodos:
  - `void addFrame(CalibrationFrame f)`
  - `void removeAt(int index)`
  - `void clear()`
  - `int size()`
  - `List<CalibrationFrame> getAll()` (cópia defensiva)
- Notifica observers via callback ou `LiveData` interno.

#### 3.3 `repository/CalibrationJsonStore.java`
- Métodos estáticos:
  - `static void save(CalibrationResult r, Context ctx)` — salva em `Pictures/VisionProject/calibration.json` E em `getFilesDir()/calibration.json` (cache interno para uso por outros módulos).
  - `static CalibrationResult load(Context ctx)` — lê do cache interno; retorna `null` se não existir.
- Estrutura JSON:
  ```json
  {
    "version": "1.0",
    "calibration_date": "2026-05-09T22:00:00",
    "device_model": "samsung SM-G970F",
    "android_version": "12",
    "image_size": { "width": 1920, "height": 1080 },
    "pattern": { "type": "chessboard", "cols": 9, "rows": 6, "square_size_mm": 25.0 },
    "intrinsics": { "fx": 1408.5, "fy": 1410.2, "cx": 955.8, "cy": 538.1, "skew": 0.0 },
    "distortion": { "k1": -0.31, "k2": 0.12, "p1": 0.001, "p2": -0.0005, "k3": 0.0 },
    "rms_reprojection_error_px": 0.387,
    "images_used": 22,
    "images_rejected": 5,
    "elapsed_ms_total": 4830
  }
  ```

### 🔹 Passo 4 — Pipeline de calibração de Zhang

#### 4.1 `pipeline/ZhangCalibrationPipeline.java`

**Template Method**: 5 passos fixos.

```java
public class ZhangCalibrationPipeline {
    public CalibrationResult run(List<CalibrationFrame> frames, PatternSpec pattern, Size imageSize) {
        // 1. Validate input
        if (frames.size() < 15) throw new IllegalStateException("Need >=15 frames");

        // 2. Build objectPoints / imagePoints lists
        List<Mat> objPts = new ArrayList<>(), imgPts = new ArrayList<>();
        for (CalibrationFrame f : frames) {
            objPts.add(pattern.generateObjectPoints());
            imgPts.add(f.getCorners());
        }

        // 3. Run OpenCV
        Mat K = new Mat(), D = new Mat();
        List<Mat> rvecs = new ArrayList<>(), tvecs = new ArrayList<>();
        long t0 = System.currentTimeMillis();
        double rms = Calib3d.calibrateCamera(objPts, imgPts, imageSize, K, D, rvecs, tvecs);
        long elapsed = System.currentTimeMillis() - t0;

        // 4. Per-image reprojection error
        List<Double> perImage = ReprojectionErrorAnalyzer.computePerImage(objPts, imgPts, rvecs, tvecs, K, D);

        // 5. Build immutable result
        return new CalibrationResult(rms, K, D, perImage, frames.size(), 0, elapsed, new Date(), imageSize);
    }
}
```

#### 4.2 `pipeline/ReprojectionErrorAnalyzer.java`
- `static List<Double> computePerImage(...)`:
  - Para cada imagem `i`: `Calib3d.projectPoints(objPts.get(i), rvecs.get(i), tvecs.get(i), K, D, projected)`.
  - Calcular `||projected - imgPts.get(i)||` (Norm L2 / N).
  - Retornar lista de RMS por imagem.
- `static int findWorstFrameIndex(List<Double> errors)` — retorna o índice com maior erro.

### 🔹 Passo 5 — ViewModel e State

#### 5.1 `CalibrationState.java` (enum)
```java
public enum CalibrationState {
    IDLE,                  // App aberto, antes de iniciar
    DETECTING,             // Procurando tabuleiro no preview
    POSE_STABLE,           // Tabuleiro detectado e parado, pronto pra capturar
    CAPTURED,              // Frame acabou de ser adicionado
    READY_TO_CALIBRATE,    // >=15 frames coletados, botão "Calibrar" habilitado
    CALIBRATING,           // calibrateCamera() rodando
    DONE,                  // Resultado disponível
    ERROR
}
```

#### 5.2 `CalibrationViewModel.java`
- Estende `AndroidViewModel`.
- `MutableLiveData<CalibrationState> state`
- `MutableLiveData<Integer> framesCollectedCount`
- `MutableLiveData<DetectionResult> lastDetection` (para overlay)
- `MutableLiveData<Float> lastBlurScore`
- `MutableLiveData<Boolean> autoCaptureEnabled`
- `MutableLiveData<CalibrationResult> result`
- `MutableLiveData<String> userMessage`
- Recebe via construtor (DI manual):
  - `PatternDetectionStrategy detector`
  - `BlurDetectionStrategy blurStrategy`
  - `PoseStabilityStrategy stabilityStrategy`
  - `ZhangCalibrationPipeline pipeline`
  - `CalibrationFramesRepository framesRepo`
- Métodos públicos:
  - `void onPreviewFrame(Mat rgba)` — chamado pelo `CvCameraViewListener`. Faz: gray, detect, blur check, pose stability check; se tudo OK e `autoCaptureEnabled` → `captureFrame()`.
  - `void captureFrame()` — força captura manual.
  - `void deleteFrame(int idx)`.
  - `void runCalibration()` — em background (Executor), chama `pipeline.run(...)`, atualiza `result`.
  - `void exportJson()` — chama `CalibrationJsonStore.save(result.getValue(), ctx)`.
  - `void reset()`.
- `onCleared()` — encerra logger, limpa repositório se quiser.

### 🔹 Passo 6 — Custom Views

#### 6.1 `ui/ChessboardOverlayView.java`
- Estende `View` (ou desenha sobre `JavaCameraView`).
- `setCorners(MatOfPoint2f corners)` — guarda referência.
- `setStatus(CalibrationState state)` — controla cor (cinza/amarelo/verde).
- `onDraw(Canvas)` — desenha:
  - Círculos sobre cada canto (verde se POSE_STABLE, amarelo se DETECTING, cinza se IDLE).
  - Linhas entre cantos consecutivos (esqueleto do tabuleiro).
  - Texto no topo: "X/20 frames coletados".

#### 6.2 `ui/CoverageHeatmapView.java`
- Estende `View`.
- Mantém grade 5×5 de células sobre o quadro da câmera.
- Para cada `CalibrationFrame` adicionado, marca quais células foram cobertas (centro do tabuleiro cai em qual célula).
- `onDraw(Canvas)` — pinta cada célula com cor baseada em quantos frames cobriram aquela região (gradiente vermelho→verde).
- Útil para o usuário saber se está cobrindo todos os cantos da imagem.

#### 6.3 `ui/BeforeAfterUndistortView.java`
- Layout horizontal com 2 `ImageView`.
- `setBefore(Bitmap)`, `setAfter(Bitmap)`.
- Reutilizar a `ComparisonImageView` do Módulo 2 se já estiver pronta — **não duplicar**.

### 🔹 Passo 7 — Activity

#### 7.1 `CalibrationActivity.java`
- Estende `AppCompatActivity`, implementa `CvCameraViewListener2`.
- Layout `cal_activity_calibracao.xml`:
  - Topo: TextView mostrando estado atual e contagem de frames
  - Centro: `JavaCameraView` com `ChessboardOverlayView` sobreposta + `CoverageHeatmapView` no canto inferior direito (mini-mapa)
  - Painel inferior: botões `[📷 Captura manual]` `[🔄 Auto-captura ON/OFF]` `[🗑 Limpar]` `[⚙ Calibrar]` `[💾 Exportar JSON]` `[👁 Antes/Depois]`
- `onCameraFrame(...)` → chama `viewModel.onPreviewFrame(...)`.
- Observa LiveData do ViewModel e atualiza UI.
- Quando `state == DONE`: mostra dialog/Toast com resumo: `"RMS = 0,387 px | fx=1408,5 fy=1410,2 cx=960 cy=540 | k1=-0,31"`.

### 🔹 Passo 8 — Logger CSV (REUTILIZAR estrutura do Módulo 2)

#### 8.1 `calibracao/logger/CalibrationCsvLogger.java`
- **Mesmo padrão** do `CalibrationSessionLogger` do Módulo 2 (Singleton + ExecutorService + writer reutilizado).
- Arquivo: `Pictures/VisionProject/cal_session_<yyyyMMdd_HHmmss>.csv`.
- Eventos:
  - `SESSION_START`
  - `FRAME_DETECTED` (a cada detecção bem-sucedida no preview, com blur_score e cobertura)
  - `FRAME_CAPTURED` (quando aceita e adiciona ao conjunto, com índice, blur_score, pose_stability_score, region)
  - `FRAME_REJECTED` (com motivo: "blur", "incomplete_corners", "manual_delete")
  - `CALIBRATION_STARTED`
  - `CALIBRATION_DONE` (com rms, fx, fy, cx, cy, k1, k2, p1, p2, k3, images_used, elapsed_ms)
  - `PER_IMAGE_ERROR` (uma linha por imagem usada, com index e error_px)
  - `JSON_EXPORTED` (com path)
  - `UNDISTORT_PREVIEW` (toda vez que o usuário aciona before/after)
  - `SESSION_CLOSED`
- Header CSV (extensão das colunas do Módulo 2):
  ```
  session_id,timestamp_ms,timestamp_iso,event_type,
  frame_index,corners_found,blur_score,pose_stability_score,coverage_region,
  fx,fy,cx,cy,k1,k2,p1,p2,k3,
  rms_reprojection_px,per_image_error_px,
  image_filename,image_w,image_h,
  elapsed_ms,images_used,images_rejected,
  rejection_reason,json_path,
  device_model,android_version,notes
  ```

### 🔹 Passo 9 — Integração com módulos anteriores

#### 9.1 Estender `CalibrationRepository` do Módulo 2 (sem quebrar)
- Adicionar **método novo** (não modificar existentes):
  ```java
  public synchronized void loadFromCalibrationJson(Context ctx) {
      CalibrationResult cr = CalibrationJsonStore.load(ctx);
      if (cr == null) return;  // mantém placeholders
      setIntrinsics(new CameraIntrinsics(cr.getFx(), cr.getFy(), cr.getCx(), cr.getCy()));
      setDistortion(new DistortionCoefficients.Builder()
          .k1(cr.getK1()).k2(cr.getK2())
          .p1(cr.getP1()).p2(cr.getP2()).k3(cr.getK3()).build());
  }
  ```
- Em `MainActivity.onCreate()`: chamar `CalibrationRepository.getInstance().loadFromCalibrationJson(this);` no início. Se o JSON não existir, mantém os placeholders do Módulo 2 (comportamento atual preservado).
- Em `ModeloCameraActivity.onResume()`: chamar o mesmo método para garantir que vê valores atualizados após uma calibração nova.

#### 9.2 MainActivity — adicionar botão "Calibração"
```xml
<Button android:id="@+id/btnCalibracao" android:text="Calibração de Câmera" />
```
```java
findViewById(R.id.btnCalibracao).setOnClickListener(v ->
    startActivity(new Intent(this, CalibrationActivity.class)));
```

#### 9.3 Manifest
```xml
<activity android:name=".calibracao.CalibrationActivity"
          android:label="Calibração"
          android:parentActivityName=".MainActivity"
          android:exported="false" />
```

### 🔹 Passo 10 — Testes unitários (mínimo)

Em `app/src/test/java/com/example/visionproject/calibracao/`:

#### 10.1 `ChessboardSpecTest.java`
- Default 9×6×25mm gera 54 pontos.
- `generateObjectPoints()` retorna `Mat` com 54 linhas.
- Primeiro ponto é (0,0,0), último é ((9-1)*25, (6-1)*25, 0) = (200, 125, 0).
- Construtor com cols=2 lança exceção.

#### 10.2 `ObjectPointsFactoryTest.java`
- `createPlanarGrid(9, 6, 25.0)` — 54 pontos, todos com Z=0.

#### 10.3 `BlurDetectionStrategyTest.java`
- Imagem uniforme (cinza puro) → score baixo, `isSharp = false`.
- Imagem com ruído alto → score alto, `isSharp = true`.

#### 10.4 `PoseStabilityStrategyTest.java`
- 5 detecções idênticas → estável.
- 5 detecções com ruído >2px → instável.

#### 10.5 `CalibrationResultTest.java`
- `toJson()` produz JSON válido com todas as chaves esperadas.

> Pipeline e Repository **não** precisam de teste unitário (dependem de OpenCV nativo / Android Context). Teste manual via app + verificação do CSV/JSON.

---

## 4. Dados que Precisam Estar no CSV (para o relatório)

A outra IA **deve garantir** que o CSV final contém pelo menos **uma linha de cada um destes eventos**, com os campos preenchidos:

| Evento | Campos críticos |
|---|---|
| `SESSION_START` | session_id, device_model, android_version |
| `FRAME_CAPTURED` (×N≥15) | frame_index, corners_found, blur_score, pose_stability_score, coverage_region |
| `CALIBRATION_DONE` | rms_reprojection_px, fx, fy, cx, cy, k1, k2, p1, p2, k3, images_used, elapsed_ms, image_w, image_h |
| `PER_IMAGE_ERROR` (×N) | frame_index, per_image_error_px |
| `JSON_EXPORTED` | json_path |
| `SESSION_CLOSED` | notes (total events) |

**Sem esses dados o relatório não tem conteúdo.** Validar manualmente abrindo o CSV no Excel após a sessão de calibração.

---

## 5. Critérios de Aceite (DoD)

- [ ] App compila sem warnings novos.
- [ ] **PCC continua funcionando** (smoke test).
- [ ] **ModeloCamera continua funcionando** (smoke test).
- [ ] CSVs anteriores (mc_session_*.csv) continuam sendo gerados normalmente.
- [ ] Novo botão "Calibração" abre `CalibrationActivity`.
- [ ] Tabuleiro 9×6 é detectado em tempo real no preview, com overlay verde sobre os cantos.
- [ ] Captura automática dispara quando pose está estável (5 frames consecutivos com shift < 2 px) e blur_score >= 100.
- [ ] Heatmap de cobertura mostra quais regiões da imagem já foram cobertas.
- [ ] Coletar 20 frames variando ângulo/distância/posição.
- [ ] Botão "Calibrar" executa `Calib3d.calibrateCamera()` em background.
- [ ] **RMS de reprojeção < 1.0 px** (ideal < 0.5 px) — se não atingir, recalibrar com mais imagens próximas das bordas.
- [ ] Resultado é exibido no app (RMS, fx, fy, cx, cy, k1, k2).
- [ ] Antes/depois de undistort exibe lado a lado e a diferença é visualmente perceptível (linhas mais retas).
- [ ] `calibration.json` é gerado em `Pictures/VisionProject/` e em `getFilesDir()/`.
- [ ] CSV de sessão contém todos os eventos da seção 4.
- [ ] `MainActivity.onCreate()` carrega `calibration.json` se existir; caso contrário, app continua funcionando com placeholders.
- [ ] `ModeloCameraActivity` passa a usar K e D reais carregados do JSON (visível na tela superior dela).
- [ ] Testes unitários (5 classes da seção 10) passam.

---

## 6. Anti-padrões a evitar

- ❌ **Não** rodar `calibrateCamera()` no main thread (pode demorar 5-30s).
- ❌ **Não** alocar `Mat` novo a cada frame do preview (vaza memória — reutilizar).
- ❌ **Não** desenhar diretamente sobre o frame capturado da câmera (use overlay View).
- ❌ **Não** quebrar o contrato público de `CalibrationRepository` do Módulo 2 — apenas **adicionar** método.
- ❌ **Não** salvar JSON apenas em local interno (perde rastreabilidade) ou apenas em externo (outros módulos não conseguem ler) — salvar **nos dois lugares**.
- ❌ **Não** usar `runOnUiThread` para fazer trabalho pesado — use `ExecutorService`.
- ❌ **Não** assumir resolução fixa — `imageSize` deve ser obtido do frame real da câmera, não hardcoded.
- ❌ **Não** ignorar imagens com cantos parcialmente fora do quadro — `findChessboardCorners` retorna `false` corretamente, mas validar.
- ❌ **Não** misturar `MatOfPoint2f` e `Mat` indiscriminadamente — APIs do OpenCV são exigentes com tipo.
- ❌ **Não** persistir `Mat` em LiveData (vaza). Persistir bitmap ou primitivos.

---

## 7. Critério de Sucesso da Entrega M1

A entrega M1 desta atividade é avaliada pela disciplina. **A IA implementadora deve garantir, ANTES de declarar pronto:**

1. **RMS final < 1.0 px** documentado no CSV. Se não conseguir, repetir a coleta de imagens (mais imagens, melhor cobertura, melhor iluminação).
2. **fx ≈ fy** (diferença <1%) — se diferirem muito, aspecto do pixel está sendo mal estimado.
3. **cx ≈ width/2 ± 50px** e **cy ≈ height/2 ± 50px** — se cair muito longe do centro, falta cobertura nas bordas.
4. **k1 negativo** (esperado para smartphones com barril leve). Se k1 positivo, há algo errado na calibração ou na detecção de cantos.
5. **calibration.json carregável** pelo Módulo 2 sem crash.
6. **Vídeo de demonstração** mostrando: detecção em tempo real → coleta automática → calibração → before/after.

---

## 8. Resumo executivo (cheat-sheet)

1. Cria pacote `calibracao/` com subpacotes `model/`, `strategy/`, `factory/`, `pipeline/`, `repository/`, `ui/`, `logger/`.
2. Modelos: `PatternSpec` interface + `ChessboardSpec` (9×6×25mm).
3. Strategies: `ChessboardDetectionStrategy`, `BlurDetectionStrategy` (Laplacian variance), `PoseStabilityStrategy` (5-frame queue).
4. Factory: `ObjectPointsFactory`.
5. Repository: `CalibrationFramesRepository` (Singleton).
6. JSON store: `CalibrationJsonStore` (salva em externo + interno).
7. Pipeline: `ZhangCalibrationPipeline` (Template Method com 5 etapas).
8. ViewModel + State enum.
9. Activity + 3 custom views (chessboard overlay, coverage heatmap, before/after).
10. Logger CSV reutilizando infra do Módulo 2.
11. Estende `CalibrationRepository` do Módulo 2 com `loadFromCalibrationJson()`.
12. Botão "Calibração" na MainActivity + entry no Manifest.
13. Testes unitários das 5 classes puras.
14. Smoke test: PCC funciona + ModeloCamera funciona + Calibração nova funciona end-to-end.
15. Atinge RMS < 1.0 px; valida k1 negativo; verifica que ModeloCamera passa a usar K/D reais.

---

**Autor:** Vinicius L. Alvarenga
**Versão:** 1.0
**Data:** Maio/2026
**Pré-requisitos:** ESPEC_ModeloCamera.md (v1) e ESPEC_ModeloCamera_v2_CSV.md já implementados.
