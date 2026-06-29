# ESPECIFICAÇÃO TÉCNICA — Módulo 4: Visão Estéreo com Smartphone
**Projeto:** VisionProject (Android, Java + OpenCV4Android + CameraX)
**Atividade:** Visão Robótica — Estéreo com smartphone único (UFLA, Prof. Arthur de Miranda Neto)
**Pré-requisitos:** Módulos 1 (PCC), 2 (ModeloCamera) e 3 (Calibração) **já implementados e funcionais**

---

## 0. Princípios Não-Negociáveis (LEIA TUDO ANTES DE CODAR)

1. **Zero alteração em código existente.** O pacote `pcc/`, `modelocamera/`, `calibracao/` e `shared/` continuam intactos. Qualquer mudança que afete API pública dessas classes é **proibida**.
2. **Tudo em pacote novo:** `com.example.visionproject.estereo`.
3. **Reaproveitar** o que já existe:
   - **Calibração K e D** — carregar o `calibration.json` gerado pelo Módulo 3 via `CalibrationJsonStore.load(ctx)`.
   - **Logger CSV** — usar a mesma infraestrutura `shared/csv/CsvWriter` + `CsvFormatter` + `DeviceInfoProvider` (se essas classes existirem; senão, reusar a estrutura do `CalibrationCsvLogger` do Módulo 3).
   - **FileExporter** (Módulo 2) — para salvar PNG/JPG no MediaStore.
4. **MVVM obrigatório** + **State enum** + **padrões de design** explicitados na seção 2.
5. **CameraX é nova dependência** — precisará ser adicionada ao `build.gradle` (não estava nos módulos anteriores). Lista completa na seção 1.3.
6. **Compatibilidade Android:** mínimo API 24 (Android 7.0), recomendado API 28+.
7. **Threading:** todas as operações pesadas (ORB, RANSAC, SGBM, undistort) em `ExecutorService` — **NUNCA na main thread**.
8. **Smoke test após implementar:** PCC funciona → ModeloCamera funciona → Calibração funciona → Estereo nova funciona.
9. **Não usar opencv-contrib** (filtro WLS, SIFT). Usar somente o módulo base OpenCV4Android: `org.opencv.core`, `org.opencv.calib3d`, `org.opencv.imgproc`, `org.opencv.features2d`.
10. **Reaproveitar `calibration.json` do Módulo 3 — NÃO implementar nova Activity de calibração.** A T1 do enunciado já foi cumprida pelo Módulo 3.

---

## 1. Estrutura de Pacotes e Dependências

### 1.1 Pacotes novos

```
com.example.visionproject/
├── pcc/                              [JÁ EXISTE — NÃO MEXER]
├── modelocamera/                     [JÁ EXISTE — NÃO MEXER]
├── calibracao/                       [JÁ EXISTE — NÃO MEXER]
├── shared/                           [JÁ EXISTE — REUSAR]
└── estereo/                          [NOVO — todo o trabalho aqui]
    ├── EstereoMainActivity.java          (hub de navegação do módulo)
    ├── CaptureActivity.java              (T2: captura par I_L, I_R)
    ├── CaptureViewModel.java
    ├── ProcessingActivity.java           (T3-T7: undistort, ORB, F, retificação, SGBM)
    ├── ProcessingViewModel.java
    ├── DepthActivity.java                (T8: Z = f·B/d + .ply)
    ├── DepthViewModel.java
    ├── EstereoState.java                 (enum: IDLE, AWAIT_LEFT, AWAIT_RIGHT, PAIR_READY, PROCESSING, DONE, ERROR)
    ├── model/
    │   ├── StereoPair.java               (par L+R + metadados)
    │   ├── FeatureMatches.java           (kpL, kpR, good matches, undL, undR)
    │   ├── FundamentalResult.java        (F, inliers, máscara RANSAC)
    │   ├── RectifiedPair.java            (rectL, rectR, H1, H2)
    │   ├── DisparityResult.java          (disp32f, params usados, tempo)
    │   ├── DepthResult.java              (Z map, métricas)
    │   └── SgbmParams.java               (immutable, com Builder)
    ├── strategy/
    │   ├── FeatureDetectionStrategy.java (interface)
    │   ├── OrbFeatureStrategy.java
    │   ├── DisparityStrategy.java        (interface)
    │   ├── SgbmDisparityStrategy.java
    │   └── BmDisparityStrategy.java      (fallback, opcional)
    ├── factory/
    │   └── SgbmParamsFactory.java        (3 presets: FAST, BALANCED, QUALITY)
    ├── pipeline/
    │   ├── StereoPipeline.java           (Template Method: undistort→match→F→rectify→disparity→post→depth)
    │   ├── FundamentalEstimator.java     (8pt + RANSAC)
    │   ├── RectificationEstimator.java   (stereoRectifyUncalibrated)
    │   ├── DisparityComputer.java        (orquestra DisparityStrategy)
    │   └── DepthTriangulator.java        (Z = f·B/d)
    ├── repository/
    │   ├── StereoPairRepository.java     (Singleton, mantém ultimo par)
    │   └── StereoCsvLogger.java          (Singleton, mesma infra dos Módulos 2-3)
    ├── ui/
    │   ├── PairPreviewView.java          (mostra L | R lado a lado)
    │   ├── EpipolarLinesView.java        (overlay de linhas epipolares coloridas)
    │   ├── RectifiedView.java            (par retificado + linhas horizontais)
    │   ├── DisparityColorMapView.java    (mapa COLORMAP_TURBO)
    │   └── DepthMeasureView.java         (clique no ponto → mostra Z em metros)
    └── export/
        ├── PlyExporter.java              (T8 bônus: .ply ASCII)
        └── ImageExporter.java            (PNG dos artefatos)
```

### 1.2 Resources (recursos)

```
res/
├── layout/
│   ├── est_activity_main.xml
│   ├── est_activity_capture.xml
│   ├── est_activity_processing.xml
│   ├── est_activity_depth.xml
│   └── est_dialog_sgbm_params.xml
├── values/
│   └── est_strings.xml
└── color/
    └── est_button_text.xml + est_button_bg.xml (reusar padrão das v3 corrections)
```

### 1.3 Dependências a adicionar em `app/build.gradle`

```gradle
def cameraxVersion = '1.3.0'

dependencies {
    // ... existentes
    implementation "androidx.camera:camera-core:${cameraxVersion}"
    implementation "androidx.camera:camera-camera2:${cameraxVersion}"
    implementation "androidx.camera:camera-lifecycle:${cameraxVersion}"
    implementation "androidx.camera:camera-view:${cameraxVersion}"
    // OpenCV já está no projeto desde Módulo 1
}
```

⚠️ Confirmar que `minSdk >= 21` para CameraX. Se atual for menor, ajustar.

### 1.4 Manifest

```xml
<!-- já tem CAMERA permission desde Módulo 1 -->
<uses-feature android:name="android.hardware.camera2" android:required="false" />

<application ...>
    <!-- registrar 4 Activities novas -->
    <activity android:name=".estereo.EstereoMainActivity"
              android:label="@string/est_title_main"
              android:exported="false"
              android:screenOrientation="portrait"
              android:parentActivityName=".MainActivity"
              android:theme="@style/Theme.VisionProject.NoWindowBackground" />
    <activity android:name=".estereo.CaptureActivity"
              android:label="@string/est_title_capture"
              android:exported="false"
              android:screenOrientation="portrait"
              android:parentActivityName=".estereo.EstereoMainActivity"
              android:theme="@style/Theme.VisionProject.NoWindowBackground" />
    <activity android:name=".estereo.ProcessingActivity"
              android:label="@string/est_title_processing"
              android:exported="false"
              android:parentActivityName=".estereo.EstereoMainActivity"
              android:theme="@style/Theme.VisionProject.NoWindowBackground" />
    <activity android:name=".estereo.DepthActivity"
              android:label="@string/est_title_depth"
              android:exported="false"
              android:parentActivityName=".estereo.EstereoMainActivity"
              android:theme="@style/Theme.VisionProject.NoWindowBackground" />
</application>
```

### 1.5 Integração com MainActivity

**Adicionar UM botão novo** (não remover nada):
```xml
<Button android:id="@+id/btn_estereo"
        android:text="@string/btn_estereo"
        ... />
```
```java
findViewById(R.id.btn_estereo).setOnClickListener(v ->
    startActivity(new Intent(this, com.example.visionproject.estereo.EstereoMainActivity.class)));
```

---

## 2. Padrões de Design

| Padrão | Onde | Por quê |
|---|---|---|
| **MVVM** | toda Activity | Separação UI / lógica / dados |
| **State** | `EstereoState` enum | UI reflete claramente fase do pipeline |
| **Strategy** | `FeatureDetectionStrategy`, `DisparityStrategy` | Trocar ORB↔SIFT, SGBM↔BM sem mudar consumidor |
| **Singleton (Holder)** | `StereoPairRepository`, `StereoCsvLogger` | Estado global thread-safe, sobrevive a rotação |
| **Factory** | `SgbmParamsFactory` | Presets FAST/BALANCED/QUALITY |
| **Builder** | `SgbmParams.Builder` | 9+ parâmetros do SGBM |
| **Template Method** | `StereoPipeline` | 7 etapas fixas (undistort, ORB, F, rectify, SGBM, post, depth) |
| **Repository** | `StereoPairRepository` | Abstrai persistência do par capturado |
| **Observer (LiveData)** | ViewModels | Reatividade |

---

## 3. Implementação ordenada (Tarefas T1-T8)

### 🔵 T1 — Calibração intrínseca

**Status:** ✅ **JÁ IMPLEMENTADA NO MÓDULO 3.**

**Ação:** **NÃO duplicar.** Reusar via:
```java
CalibrationResult cr = CalibrationJsonStore.load(ctx);
if (cr == null) {
    // mostrar diálogo: "Calibração não encontrada. Execute o Módulo 3 primeiro."
    return;
}
Mat K = cr.getCameraMatrix();
Mat distCoeffs = cr.getDistCoeffs();
```

`EstereoMainActivity` deve, no `onCreate()`, verificar se `calibration.json` existe. Se não, exibir botão "Ir para Calibração" que abre `CalibrationActivity` do Módulo 3.

### 🔵 T2 — Captura do par estéreo

#### Passos da implementação

**2.1 `CaptureActivity.java`**

Layout (`est_activity_capture.xml`):
- `PreviewView` do CameraX (área principal)
- Topo: TextView de status ("POSICIONE NA POSIÇÃO 1 (ESQUERDA)" / "AGORA POSIÇÃO 2 (DIREITA)" / "PAR COMPLETO")
- Topo: TextView baseline atual (selecionável por dialog)
- Bottom: botões `[📷 Capturar L]` `[📷 Capturar R]` `[↩ Limpar]` `[➡ Processar]`

**2.2 Configuração CameraX**

```java
private void setupCamera() {
    ListenableFuture<ProcessCameraProvider> future = ProcessCameraProvider.getInstance(this);
    future.addListener(() -> {
        try {
            cameraProvider = future.get();
            Preview preview = new Preview.Builder()
                .setTargetResolution(new Size(1280, 720))  // ou maior se disponível
                .build();
            preview.setSurfaceProvider(previewView.getSurfaceProvider());

            imageCapture = new ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
                .setTargetResolution(new Size(1280, 720))
                .build();

            CameraSelector selector = CameraSelector.DEFAULT_BACK_CAMERA;
            Camera camera = cameraProvider.bindToLifecycle(this, selector, preview, imageCapture);

            // Lock AF/AE/AWB para garantir intrínsecos consistentes entre L e R
            lockCameraSettings(camera);
        } catch (Exception e) { /* log + dialog */ }
    }, ContextCompat.getMainExecutor(this));
}

@SuppressLint("UnsafeOptInUsageError")
private void lockCameraSettings(Camera camera) {
    Camera2CameraControl c2 = Camera2CameraControl.from(camera.getCameraControl());
    CaptureRequestOptions opt = new CaptureRequestOptions.Builder()
        .setCaptureRequestOption(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_OFF)
        .setCaptureRequestOption(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_OFF)
        .setCaptureRequestOption(CaptureRequest.CONTROL_AWB_MODE, CaptureRequest.CONTROL_AWB_MODE_OFF)
        .build();
    c2.setCaptureRequestOptions(opt);
}
```

**2.3 Captura**

```java
private void captureImage(String side) {  // side = "L" ou "R"
    File outDir = new File(getFilesDir(), "estereo_pairs/" + sessionId);
    if (!outDir.exists()) outDir.mkdirs();
    File outFile = new File(outDir, "I_" + side + ".jpg");

    ImageCapture.OutputFileOptions opts =
        new ImageCapture.OutputFileOptions.Builder(outFile).build();

    imageCapture.takePicture(opts, ContextCompat.getMainExecutor(this),
        new ImageCapture.OnImageSavedCallback() {
            @Override public void onImageSaved(ImageCapture.OutputFileResults r) {
                viewModel.onImageSaved(side, outFile);
                StereoCsvLogger.getInstance().log("PAIR_CAPTURED_" + side, outFile.getAbsolutePath());
            }
            @Override public void onError(ImageCaptureException e) {
                Toast.makeText(CaptureActivity.this, "Erro ao capturar: " + e.getMessage(),
                    Toast.LENGTH_LONG).show();
            }
        });
}
```

**2.4 ViewModel mantém estado:**
- `EstereoState` (AWAIT_LEFT → AWAIT_RIGHT → PAIR_READY)
- `MutableLiveData<File> leftFile`, `rightFile`
- `MutableLiveData<Float> baselineMm` (default 60.0f, editável por dialog)
- Quando o par está completo: persistir no `StereoPairRepository` e habilitar botão "Processar".

**2.5 Baseline configurável**

Dialog que aparece ao clicar em um botão "⚙ Baseline":
- Slider de 40 a 150 mm (default 60)
- TextInput para digitar valor exato
- Salva em SharedPreferences (key `est.baseline_mm`)
- Loga `BASELINE_SET` no CSV

**Critério de aceite T2:**
- [ ] Foto L e R salvas em `getFilesDir()/estereo_pairs/<sessionId>/I_L.jpg` e `I_R.jpg`
- [ ] AF/AE/AWB trancados antes da captura (visível: brilho da imagem não muda entre L e R)
- [ ] Baseline registrada em SharedPrefs e exibida na tela
- [ ] CSV recebe `PAIR_CAPTURED_L` e `PAIR_CAPTURED_R` com paths absolutos
- [ ] Após captura completa, estado vai para `PAIR_READY` e botão Processar habilita

### 🔵 T3 — Pré-processamento e correspondência ORB

**Local:** `pipeline/StereoPipeline.java` etapa 1-2.

```java
// 3.1 Carregar par + undistort
Mat imgL = Imgcodecs.imread(pair.leftFile.getAbsolutePath());
Mat imgR = Imgcodecs.imread(pair.rightFile.getAbsolutePath());

Mat undL = new Mat(), undR = new Mat();
Imgproc.undistort(imgL, undL, K, distCoeffs);
Imgproc.undistort(imgR, undR, K, distCoeffs);

Mat grayL = new Mat(), grayR = new Mat();
Imgproc.cvtColor(undL, grayL, Imgproc.COLOR_BGR2GRAY);
Imgproc.cvtColor(undR, grayR, Imgproc.COLOR_BGR2GRAY);

// 3.2 ORB features
ORB orb = ORB.create(2000);
MatOfKeyPoint kpL = new MatOfKeyPoint(), kpR = new MatOfKeyPoint();
Mat desL = new Mat(), desR = new Mat();
orb.detectAndCompute(grayL, new Mat(), kpL, desL);
orb.detectAndCompute(grayR, new Mat(), kpR, desR);

// 3.3 Matching KNN + Lowe ratio test
DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);
List<MatOfDMatch> knn = new ArrayList<>();
matcher.knnMatch(desL, desR, knn, 2);

List<DMatch> good = new ArrayList<>();
for (MatOfDMatch m : knn) {
    DMatch[] arr = m.toArray();
    if (arr.length >= 2 && arr[0].distance < 0.75f * arr[1].distance)
        good.add(arr[0]);
}

// 3.4 Logger + visualização
logger.log("ORB_DONE", "kpL=" + kpL.size() + " kpR=" + kpR.size() +
                       " matches_raw=" + knn.size() + " matches_good=" + good.size());

// Desenhar matches para exibir
Mat drawn = new Mat();
MatOfDMatch goodMat = new MatOfDMatch();
goodMat.fromList(good);
Features2d.drawMatches(undL, kpL, undR, kpR, goodMat, drawn);
saveAsArtifact(drawn, "t3_matches.png");
```

**Critério de aceite T3:**
- [ ] Mostrar na `ProcessingActivity` a imagem de matches (Features2d.drawMatches) salva como PNG
- [ ] Mostrar TextView: "Matches: raw=N, good=M"
- [ ] CSV recebe `ORB_DONE` com kp counts e match counts

### 🔵 T4 — Matriz Fundamental F + linhas epipolares

**Local:** `pipeline/FundamentalEstimator.java`.

```java
public class FundamentalEstimator {
    public static FundamentalResult estimate(List<KeyPoint> kpL, List<KeyPoint> kpR,
                                             List<DMatch> good) {
        if (good.size() < 8) throw new IllegalStateException("Mín 8 matches");

        List<Point> ptsL = new ArrayList<>(), ptsR = new ArrayList<>();
        for (DMatch m : good) {
            ptsL.add(kpL.get(m.queryIdx).pt);
            ptsR.add(kpR.get(m.trainIdx).pt);
        }

        MatOfPoint2f p1 = new MatOfPoint2f();
        MatOfPoint2f p2 = new MatOfPoint2f();
        p1.fromList(ptsL); p2.fromList(ptsR);

        Mat mask = new Mat();
        Mat F = Calib3d.findFundamentalMat(p1, p2,
            Calib3d.FM_RANSAC,
            3.0,    // ransacReprojThreshold (px)
            0.99,   // confidence
            mask);

        int inliers = Core.countNonZero(mask);

        // Filtrar pontos para inliers
        List<Point> inL = new ArrayList<>(), inR = new ArrayList<>();
        for (int i = 0; i < mask.rows(); i++) {
            if (mask.get(i,0)[0] != 0) {
                inL.add(ptsL.get(i)); inR.add(ptsR.get(i));
            }
        }

        return new FundamentalResult(F, inliers, mask, inL, inR);
    }
}
```

**Visualização epipolar (UI/EpipolarLinesView):**
1. Selecionar 10 pontos aleatórios dos inliers em L.
2. Calcular linhas correspondentes em R: `Calib3d.computeCorrespondEpilines(p1, 1, F, lines)`.
3. Para cada linha (a,b,c), traçar em R: `y = (-c - a*x) / b` para x=0 e x=cols-1.
4. Usar 10 cores distintas (rainbow). Desenhar círculo no ponto correspondente em L.
5. Exibir imagens L+R lado a lado com overlays.

**Critério de aceite T4:**
- [ ] F estimada (Mat 3×3)
- [ ] Inliers > 30 (se menor, mostrar warning na UI)
- [ ] Par L+R exibido com 10 linhas epipolares coloridas que **passam visualmente próximas** dos pontos correspondentes
- [ ] CSV recebe `F_ESTIMATED` com inlier_count

### 🔵 T5 — Retificação (não calibrada)

**Local:** `pipeline/RectificationEstimator.java`.

```java
public static RectifiedPair rectify(Mat undL, Mat undR, FundamentalResult fr) {
    Size size = undL.size();
    MatOfPoint2f inL = new MatOfPoint2f();
    MatOfPoint2f inR = new MatOfPoint2f();
    inL.fromList(fr.inliersL); inR.fromList(fr.inliersR);

    Mat H1 = new Mat(), H2 = new Mat();
    boolean ok = Calib3d.stereoRectifyUncalibrated(inL, inR, fr.F, size, H1, H2, 5.0);
    if (!ok) throw new RuntimeException("stereoRectifyUncalibrated falhou");

    Mat rectL = new Mat(), rectR = new Mat();
    Imgproc.warpPerspective(undL, rectL, H1, size);
    Imgproc.warpPerspective(undR, rectR, H2, size);

    return new RectifiedPair(rectL, rectR, H1, H2);
}
```

**Visualização (UI/RectifiedView):**
- Mostra par retificado lado a lado.
- Traça **10 linhas horizontais** (cores rotativas) atravessando ambas as imagens, em y = 50, 150, 250, ..., 950.
- Permite ao usuário verificar visualmente que objetos correspondentes estão na mesma altura.

**Critério de aceite T5:**
- [ ] Par retificado salvo como PNG (`t5_rectified.png`)
- [ ] Linhas horizontais sobrepostas → objetos correspondentes (ex.: cantos de mesa) aparecem na mesma linha em L e R
- [ ] Caso `stereoRectifyUncalibrated` falhe (return false), mostrar dialog: "Retificação falhou — verifique a captura"

### 🔵 T6 — Mapa de disparidade (SGBM)

**Local:** `pipeline/DisparityComputer.java` + `strategy/SgbmDisparityStrategy.java`.

```java
public class SgbmDisparityStrategy implements DisparityStrategy {
    private final SgbmParams params;

    public SgbmDisparityStrategy(SgbmParams p) { this.params = p; }

    @Override
    public DisparityResult compute(Mat grayL, Mat grayR) {
        long t0 = System.currentTimeMillis();

        StereoSGBM sgbm = StereoSGBM.create(
            params.minDisparity,
            params.numDisparities,    // múltiplo de 16
            params.blockSize,
            params.P1,                // 8 * 3 * blockSize^2
            params.P2,                // 32 * 3 * blockSize^2
            params.disp12MaxDiff,
            params.preFilterCap,
            params.uniquenessRatio,
            params.speckleWindowSize,
            params.speckleRange,
            StereoSGBM.MODE_SGBM
        );

        Mat disp16 = new Mat();
        sgbm.compute(grayL, grayR, disp16);

        Mat disp32 = new Mat();
        disp16.convertTo(disp32, CvType.CV_32F, 1.0/16.0);

        long elapsed = System.currentTimeMillis() - t0;
        return new DisparityResult(disp32, params, elapsed);
    }
}
```

**3 presets via Factory:**

```java
public class SgbmParamsFactory {
    public static SgbmParams fast() {
        return new SgbmParams.Builder()
            .numDisparities(64).blockSize(5)
            .uniquenessRatio(10).speckleWindowSize(100).speckleRange(2)
            .build();
    }
    public static SgbmParams balanced() {
        return new SgbmParams.Builder()
            .numDisparities(128).blockSize(5)
            .uniquenessRatio(10).speckleWindowSize(100).speckleRange(2)
            .build();
    }
    public static SgbmParams quality() {
        return new SgbmParams.Builder()
            .numDisparities(192).blockSize(7)
            .uniquenessRatio(15).speckleWindowSize(150).speckleRange(2)
            .build();
    }
}
```

**Visualização colormap:**
```java
Mat disp8 = new Mat();
Core.normalize(dr.disp32f, disp8, 0, 255, Core.NORM_MINMAX, CvType.CV_8U);
Mat colored = new Mat();
Imgproc.applyColorMap(disp8, colored, Imgproc.COLORMAP_TURBO);
```

**Critério de aceite T6:**
- [ ] Botões na UI para alternar entre 3 presets (FAST / BALANCED / QUALITY)
- [ ] Cada preset salva PNG `t6_disp_<preset>.png` em colormap
- [ ] CSV recebe 3 `SGBM_RUN` (um por preset) com `elapsed_ms`, params, e métricas (% pixels válidos)

### 🔵 T7 — Pós-processamento + análise

**Local:** `pipeline/StereoPipeline.java` etapa 6.

```java
// Filtro mediano (sem opencv-contrib)
Mat disp8filt = new Mat();
Imgproc.medianBlur(disp8, disp8filt, 5);   // janela 5x5

// Salvar antes/depois lado a lado
Mat sideBySide = new Mat();
Core.hconcat(Arrays.asList(disp8, disp8filt), sideBySide);
```

**Análise crítica (gerada pelo app + log no CSV):**

Para cada cena processada, calcular e logar:
- % de pixels com disparidade válida (disp > 0): `Core.countNonZero(disp8) / total`
- Disparidade média e desvio padrão na área central da imagem
- Razão sinal/ruído estimada (variância antes vs depois do filtro)
- Identificar regiões "buracadas" (sem textura): se >30% dos pixels são inválidos no preset BALANCED, registrar como `LOW_TEXTURE_SCENE`

**Critério de aceite T7:**
- [ ] Antes/depois do `medianBlur` exibido lado a lado em PNG
- [ ] CSV recebe `POSTPROCESS_DONE` com pct_valid_before e pct_valid_after

### 🔵 T8 — Profundidade Z + reconstrução 3D

**Local:** `pipeline/DepthTriangulator.java`.

```java
public class DepthTriangulator {
    public static Mat triangulate(Mat disp32f, double fxPixels, double baselineMeters) {
        Mat depth = new Mat(disp32f.size(), CvType.CV_32F);
        for (int y = 0; y < disp32f.rows(); y++) {
            for (int x = 0; x < disp32f.cols(); x++) {
                double d = disp32f.get(y, x)[0];
                if (d > 0.5) {
                    depth.put(y, x, (float)(fxPixels * baselineMeters / d));
                } else {
                    depth.put(y, x, 0f);   // inválido
                }
            }
        }
        return depth;
    }
}
```

**UI/DepthActivity:**
- Exibe disparidade (colormap) em ImageView clicável
- Ao usuário tocar em um ponto: mostra Toast: `"Pixel (x,y) → Z = X.XX m"`
- TextView fixo no topo: `"f=711 px · B=60 mm · escala válida"`
- Botão **"📏 Comparar com referência"**: dialog onde usuário digita a distância real medida com fita métrica (em metros). App calcula `erro_relativo = |Z_medido - Z_real| / Z_real * 100` e mostra.

**Exportar .ply (bônus):**

```java
public class PlyExporter {
    public static File exportAscii(Mat depth, Mat colorBGR, double fx, double cx, double cy,
                                    File outFile) {
        try (PrintWriter w = new PrintWriter(outFile)) {
            int count = countValid(depth);
            w.println("ply");
            w.println("format ascii 1.0");
            w.println("element vertex " + count);
            w.println("property float x");
            w.println("property float y");
            w.println("property float z");
            w.println("property uchar red");
            w.println("property uchar green");
            w.println("property uchar blue");
            w.println("end_header");
            for (int y = 0; y < depth.rows(); y++) {
                for (int x = 0; x < depth.cols(); x++) {
                    float z = (float) depth.get(y, x)[0];
                    if (z <= 0 || Float.isNaN(z) || Float.isInfinite(z)) continue;
                    float X = (float) ((x - cx) * z / fx);
                    float Y = (float) ((y - cy) * z / fx);
                    double[] bgr = colorBGR.get(y, x);
                    w.printf(Locale.US, "%.4f %.4f %.4f %d %d %d%n",
                        X, Y, z, (int)bgr[2], (int)bgr[1], (int)bgr[0]);
                }
            }
        }
        return outFile;
    }
}
```

**Critério de aceite T8:**
- [ ] Tela exibe mapa de disparidade clicável; toque mostra `Z` em metros
- [ ] Botão de comparação com referência funciona e mostra erro relativo
- [ ] Arquivo `.ply` salvo em `Pictures/VisionProject/estereo_<sessionId>.ply`
- [ ] CSV recebe `DEPTH_DONE` com fx, baseline, sample Z values e erro

---

## 4. Logger CSV — estrutura

**Arquivo:** `Documents/VisionProject/est_session_<yyyyMMdd_HHmmss>.csv` (mesmo padrão dos módulos anteriores)

**Header:**
```
session_id,timestamp_ms,timestamp_iso,event_type,
scene_id,side,baseline_mm,
matches_raw,matches_good,inliers_ransac,
sgbm_preset,num_disparities,block_size,elapsed_ms,pct_valid,
fx_px,reference_distance_m,measured_z_m,error_pct,
file_path,
device_model,android_version,notes
```

**Eventos:**
- `SESSION_START`, `SESSION_CLOSED`
- `BASELINE_SET` (baseline_mm)
- `PAIR_CAPTURED_L`, `PAIR_CAPTURED_R` (file_path)
- `CALIBRATION_LOADED` (fx_px)
- `UNDISTORT_DONE`
- `ORB_DONE` (matches_raw, matches_good)
- `F_ESTIMATED` (inliers_ransac)
- `RECTIFY_DONE` ou `RECTIFY_FAILED`
- `SGBM_RUN` (sgbm_preset, params, elapsed_ms, pct_valid) — uma linha por preset
- `POSTPROCESS_DONE` (pct_valid_before, pct_valid_after)
- `DEPTH_QUERY` (pixel x,y → Z medido em m)
- `REFERENCE_COMPARED` (reference_distance_m, measured_z_m, error_pct)
- `PLY_EXPORTED` (file_path, vertex_count)

---

## 5. EstereoMainActivity — hub de navegação

```
┌──────────────────────────────────────┐
│  Módulo 4 — Visão Estéreo            │
├──────────────────────────────────────┤
│  Status da calibração:               │
│  [✓] Calibração carregada            │
│     fx = 711 px, RMS = 0,50          │
│   OU                                 │
│  [✗] Calibração não encontrada       │
│     [Ir para Módulo 3 - Calibrar]    │
├──────────────────────────────────────┤
│  Baseline atual: 60 mm  [⚙ Editar]   │
├──────────────────────────────────────┤
│  Fluxo do experimento:               │
│                                      │
│  1. [📷 Capturar par]                 │
│  2. [⚙ Processar par]    [desabilita │
│                            se não    │
│                            houver par]│
│  3. [📏 Medir profundidade]                                       │
│                                      │
│  [← Voltar]                          │
└──────────────────────────────────────┘
```

---

## 6. Critérios de Aceite (DoD consolidado)

### Regressão (smoke test obrigatório)
- [ ] PCC continua funcionando
- [ ] ModeloCamera continua funcionando
- [ ] Calibração (Módulo 3) continua funcionando
- [ ] Nenhum CSV de módulos anteriores quebrou

### T2 — Captura
- [ ] Pelo menos **3 cenas** capturadas (controlada, média, natural)
- [ ] AF/AE/AWB trancados (confirmar visualmente que brilho não muda entre L e R)
- [ ] Baseline configurável e persistida

### T3-T4 — Matching e F
- [ ] Pelo menos **30 inliers RANSAC** por cena
- [ ] Linhas epipolares **passam visualmente próximas** dos pontos correspondentes

### T5 — Retificação
- [ ] Par retificado com linhas horizontais cruzando objetos correspondentes na mesma altura

### T6 — Disparidade
- [ ] **3 presets SGBM** rodados e comparados
- [ ] Mapa de disparidade com **estrutura reconhecível** (objetos do primeiro plano mais "quentes" que fundo)

### T7 — Pós-processamento
- [ ] Antes/depois `medianBlur` lado a lado
- [ ] Discussão das fontes de erro registrada em campo "notes" do CSV

### T8 — Profundidade
- [ ] Clique em ponto da disparidade retorna Z em metros
- [ ] Comparação com fita métrica (pelo menos 1 ponto por cena)
- [ ] Erro relativo discutido na UI e no CSV

### Persistência e integração
- [ ] CSV de sessão completo gerado
- [ ] Arquivos `.ply` exportados (T8 bônus)
- [ ] `calibration.json` do Módulo 3 carregado automaticamente

---

## 7. Anti-padrões a evitar

- ❌ **Não** rodar SGBM/ORB/RANSAC na main thread (usar `ExecutorService` single-thread)
- ❌ **Não** misturar OpenCV `JavaCameraView` (módulos anteriores) com `CameraX` (este módulo) na mesma Activity
- ❌ **Não** quebrar API pública de qualquer classe pré-existente
- ❌ **Não** depender de opencv-contrib (SIFT, WLS, ximgproc) — usar somente o módulo base
- ❌ **Não** salvar `Mat` em LiveData (vaza memória) — usar Bitmap ou paths
- ❌ **Não** assumir resolução de captura fixa — sempre obter via `Imgcodecs.imread(...).size()`
- ❌ **Não** alocar matrizes intermediárias dentro de loops apertados — pre-alocar fora
- ❌ **Não** esquecer de chamar `.release()` em Mats criadas explicitamente quando não estão mais em uso
- ❌ **Não** usar Bitmaps de altíssima resolução direto na UI sem downscale (causa OOM)
- ❌ **Não** confundir Mat BGR vs RGB ao exibir em ImageView — converter com `Imgproc.cvtColor`

---

## 8. Estrutura de smoke test final (você roda)

1. **PCC (Módulo 1):** abrir → gravar 30s → CSV gerado normalmente
2. **ModeloCamera (Módulo 2):** abrir → ver K e D reais carregados → capturar foto → ver comparação
3. **Calibração (Módulo 3):** abrir → ver `calibration.json` carregado se existir
4. **Estereo (Módulo 4 — NOVO):**
   - Abrir → ver "Calibração carregada: fx=711 RMS=0.50"
   - Editar baseline → 60 mm
   - Capturar L → mover phone → capturar R
   - Processar → ver matches → F → retificação → 3 disparidades
   - Tocar em ponto → ver Z
   - Comparar com referência (medir com fita métrica algum objeto, digitar valor)
   - Exportar .ply
5. **Verificar:**
   - CSV `est_session_*.csv` em `Documents/VisionProject/` com todos os eventos
   - Artefatos PNG em `Pictures/VisionProject/estereo/` (matches, epipolar, rectified, 3×disparity, before/after, depth)
   - Arquivo `.ply` em `Pictures/VisionProject/`

---

## 9. Cheat-sheet de execução

1. Adicionar dependências CameraX no `build.gradle` e sincronizar
2. Criar pacote `estereo/` com 9 subpacotes
3. Implementar modelos (`StereoPair`, `SgbmParams` com Builder, etc.)
4. Implementar strategies (`OrbFeatureStrategy`, `SgbmDisparityStrategy`)
5. Implementar pipeline (`StereoPipeline` + componentes individuais)
6. Implementar repositórios e logger (Singletons)
7. Implementar custom views
8. Implementar `CaptureActivity` com CameraX + lock AF/AE
9. Implementar `ProcessingActivity` orquestrando o pipeline
10. Implementar `DepthActivity` com clique + comparação
11. Implementar `EstereoMainActivity` (hub)
12. Adicionar Activities ao Manifest
13. Adicionar botão na MainActivity
14. Adicionar strings em `est_strings.xml`
15. Smoke test completo (seção 8)

---

## 10. O que NÃO está no escopo desta atividade

- ❌ Calibração nova (já feita no Módulo 3 — apenas REUSAR)
- ❌ SIFT / SURF (exige opencv-contrib)
- ❌ Filtro WLS de disparidade (exige opencv-contrib)
- ❌ Stereo BM (incluir como Strategy alternativa **opcional**, foco em SGBM)
- ❌ Calibração estéreo (`stereoCalibrate`) — usa retificação **não calibrada** via F estimada
- ❌ Apresentação .pptx (entrega separada — eu posso fazer depois com seus dados)
- ❌ Relatório PDF (entrega separada — eu posso fazer depois com seus dados)
- ❌ Vídeo YouTube (você grava)
- ❌ Setup físico (você constrói — só fotos no relatório)

---

**Autor da especificação:** Vinicius L. Alvarenga
**Versão:** 1.0
**Data:** Maio/2026
**Pré-requisitos:** Módulos 1, 2 e 3 já implementados e funcionais.
