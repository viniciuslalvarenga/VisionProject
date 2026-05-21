# ESPECIFICAÇÃO TÉCNICA — Activity `ModeloCamera`
**Projeto:** VisionProject (Android, Java + OpenCV4Android)
**Módulo:** 2 — Formação da Imagem e Modelo de Câmera
**Pré-requisito:** Módulo 1 (PCC) já implementado e funcional — **NÃO PODE SER ALTERADO**

---

## 0. Princípios Não-Negociáveis

Antes de escrever uma linha de código, leia toda esta seção.

1. **Isolamento total do código existente.** O pacote `com.example.visionproject.pcc` (PccModule, PccActivity, e qualquer classe relacionada ao Módulo 1) **não pode ser modificado**. Toda a lógica do PCC precisa continuar funcionando exatamente como hoje. Se sentir necessidade de alterar alguma classe do PCC, **pare e reporte** ao invés de mudar.
2. **Tudo em pacote novo isolado.** Criar `com.example.visionproject.modelocamera` com toda a nova lógica.
3. **MVVM obrigatório.** UI (Activity) só faz binding e callbacks; lógica fica em ViewModel; modelo (parâmetros, undistort) fica em classes de domínio.
4. **OpenCV via JNI já está integrado** (Módulo 1 usa). **Reaproveitar** o mesmo `OpenCVLoader.initDebug()` da MainActivity. Não criar segunda inicialização.
5. **Recursos isolados.** Layouts, strings, drawables novos vão em arquivos prefixados com `mc_` (de "ModeloCamera"). Ex.: `mc_activity_modelo_camera.xml`, `mc_strings.xml`. Não jogar tudo no `strings.xml` global.
6. **Permissões.** Já existem CAMERA e STORAGE para o Módulo 1. **Não duplicar pedidos**. Reaproveitar o helper de permissão existente, se houver.
7. **Navegação.** Adicionar **um novo botão na MainActivity** (ou menu) chamado "Modelo de Câmera" que faz `startActivity(Intent(this, ModeloCameraActivity::class.java))`. **Não substituir** o botão de PCC.

---

## 1. Estrutura de Pacotes

```
com.example.visionproject/
├── pcc/                              [JÁ EXISTE — NÃO MEXER]
│   ├── PccModule.java
│   └── PccActivity.java
├── modelocamera/                     [NOVO — todo o trabalho aqui]
│   ├── ModeloCameraActivity.java     (UI principal)
│   ├── ModeloCameraViewModel.java    (lógica de orquestração)
│   ├── model/
│   │   ├── CameraIntrinsics.java     (encapsula matriz K)
│   │   ├── DistortionCoefficients.java (encapsula k1,k2,p1,p2,k3)
│   │   └── EpipolarPoint.java        (par (Point, Color))
│   ├── strategy/                     (Strategy pattern)
│   │   ├── UndistortStrategy.java    (interface)
│   │   ├── PointUndistortStrategy.java
│   │   └── ImageUndistortStrategy.java
│   ├── factory/
│   │   └── IntrinsicsFactory.java    (Factory pattern)
│   ├── repository/
│   │   └── CalibrationRepository.java (Singleton, fornece K e D)
│   └── ui/
│       ├── ComparisonImageView.java  (custom view: original | corrigida)
│       └── EpipolarOverlayView.java  (custom view: 5 pontos + linhas)
└── shared/                           [NOVO mas reutilizável]
    ├── ImageCaptureHelper.java       (abstrai captura/galeria)
    └── FileExporter.java             (salva PNG/JPEG no MediaStore)
```

---

## 2. Padrões de Design Utilizados

| Padrão | Onde | Por quê |
|---|---|---|
| **MVVM** | Toda a Activity | Separação clara UI / lógica / dados |
| **Strategy** | `UndistortStrategy` | Permitir aplicar undistort em ponto único OU imagem inteira sem trocar a classe consumidora |
| **Singleton** | `CalibrationRepository` | Parâmetros K e D são globais ao app, devem ser únicos e thread-safe |
| **Factory** | `IntrinsicsFactory` | Criar K para diferentes resoluções (1080p, 4K) sem expor construção interna |
| **Observer** (LiveData) | ViewModel ↔ Activity | Reatividade nativa Android |
| **Builder** (opcional) | `DistortionCoefficients.Builder` | Construir coeficientes sem ambiguidade de parâmetros |

---

## 3. Lista Ordenada de Implementação

A ordem **importa** — cada item depende dos anteriores. Não pular passos.

### 🔹 Passo 1 — Modelos de domínio (sem dependência de Android)

#### 1.1 `model/CameraIntrinsics.java`
- Classe imutável (campos `final`).
- Campos: `fx`, `fy`, `cx`, `cy` (double), `s` (skew, default 0.0).
- Construtor recebe os 4 (ou 5) valores; valida que `fx > 0`, `fy > 0`.
- Método `toMat()` que retorna `org.opencv.core.Mat` 3×3 CV_64F com a matriz K.
- Método `getHorizontalFOVDegrees(int imageWidth)` que retorna FOV horizontal calculado.
- Método `getResolutionEstimate()` que retorna `Size(2*cx, 2*cy)`.
- `equals()`, `hashCode()`, `toString()`.

#### 1.2 `model/DistortionCoefficients.java`
- Classe imutável.
- Campos: `k1`, `k2`, `p1`, `p2`, `k3` (todos double).
- Construtor com Builder pattern (5 parâmetros pode confundir):
  ```java
  DistortionCoefficients d = new DistortionCoefficients.Builder()
      .k1(0.12).k2(-0.08).p1(0.001).p2(0.0005).k3(0.0).build();
  ```
- Método `toMat()` retorna `MatOfDouble` com 5 valores (ordem k1,k2,p1,p2,k3 — **não** mudar essa ordem; OpenCV exige).
- Método `getDistortionType()` retorna enum `BARREL`, `PINCUSHION` ou `NONE` baseado no sinal de k1 (k1 < -0.05 → BARREL, k1 > 0.05 → PINCUSHION, senão NONE).

#### 1.3 `model/EpipolarPoint.java`
- POJO simples: `Point coords`, `int colorRGB`, `int index` (0–4).
- Imutável.

### 🔹 Passo 2 — Repository Singleton (parâmetros globais)

#### 2.1 `repository/CalibrationRepository.java`
- **Singleton** thread-safe (initialization-on-demand holder pattern):
  ```java
  public class CalibrationRepository {
      private CalibrationRepository() {}
      private static class Holder {
          static final CalibrationRepository INSTANCE = new CalibrationRepository();
      }
      public static CalibrationRepository getInstance() { return Holder.INSTANCE; }
      // ...
  }
  ```
- Mantém referência atual de `CameraIntrinsics` e `DistortionCoefficients`.
- Métodos: `getIntrinsics()`, `getDistortion()`, `setIntrinsics(...)`, `setDistortion(...)`.
- Inicializa com valores **padrão** do material da disciplina:
  - K para 1920×1080 com fx=fy=1408.5, cx=955.8, cy=538.1 (ou os atuais do app, conforme screenshot)
  - D = [0.12, -0.08, 0.001, 0.0005, 0.0]

### 🔹 Passo 3 — Factory para diferentes resoluções

#### 3.1 `factory/IntrinsicsFactory.java`
- Métodos estáticos:
  - `static CameraIntrinsics defaultFor1080p()`
  - `static CameraIntrinsics defaultFor4K()`
  - `static CameraIntrinsics fromResolution(int w, int h, double fovHorizontalDegrees)`
- A ideia é **não hardcodar** os valores na Activity — sempre obter via Factory.

### 🔹 Passo 4 — Strategy de Undistort

#### 4.1 `strategy/UndistortStrategy.java` (interface)
```java
public interface UndistortStrategy<INPUT, OUTPUT> {
    OUTPUT undistort(INPUT input, CameraIntrinsics K, DistortionCoefficients D);
}
```

#### 4.2 `strategy/PointUndistortStrategy.java`
- Implementa `UndistortStrategy<Point, Point>`.
- Internamente chama `Calib3d.undistortPoints(...)` (já presente na implementação atual conforme screenshot).
- **Reaproveitar** o código existente que faz a correção de ponto (1200,800) → (1198.06, 797.87) — **encapsular** dentro desta strategy.

#### 4.3 `strategy/ImageUndistortStrategy.java`
- Implementa `UndistortStrategy<Mat, Mat>`.
- Internamente chama `Calib3d.undistort(srcMat, dstMat, K.toMat(), D.toMat())`.
- Cuidado: precisa **alocar** `dstMat = new Mat()` antes da chamada.
- Não fazer I/O dentro da strategy — só transformação. Ler/salvar arquivo é responsabilidade do `FileExporter`.

### 🔹 Passo 5 — Helpers compartilhados (em `shared/`)

#### 5.1 `shared/ImageCaptureHelper.java`
- Encapsula 2 fluxos de obtenção de imagem:
  - `void launchCameraCapture(ActivityResultLauncher<Intent> launcher)` — abre app de câmera nativa, retorna URI.
  - `void launchGalleryPicker(ActivityResultLauncher<Intent> launcher)` — abre galeria.
- Método utilitário `static Mat uriToMat(Context ctx, Uri uri)` que carrega imagem do URI e converte para OpenCV Mat (RGB).
- **Não** acoplar com nenhuma Activity específica — recebe `Context` por parâmetro.

#### 5.2 `shared/FileExporter.java`
- `static Uri saveMatAsPng(Context ctx, Mat image, String filename)` — salva no diretório `Pictures/VisionProject/` via MediaStore (Android Q+).
- `static Uri saveBitmapAsPng(...)` — sobrecarga.
- `static void shareImage(Context ctx, Uri imageUri)` — gera Intent.ACTION_SEND para compartilhar a imagem.

### 🔹 Passo 6 — Custom Views (UI)

#### 6.1 `ui/ComparisonImageView.java`
- Estende `androidx.appcompat.widget.AppCompatImageView` ou cria `LinearLayout` horizontal.
- Exibe duas imagens lado a lado (original | corrigida) com label sobre cada uma.
- Métodos: `setOriginal(Bitmap)`, `setCorrected(Bitmap)`.
- Suporta zoom/pan opcional (use `PhotoView` lib se quiser).

#### 6.2 `ui/EpipolarOverlayView.java`
- Estende `View`.
- Exibe uma imagem de fundo (a imagem capturada, escalada para caber).
- Implementa `onTouchEvent(MotionEvent)`:
  - Cada toque adiciona um `EpipolarPoint` à lista (até 5).
  - Cada ponto recebe cor diferente (ROYGB).
  - Toque após o 5º **substitui** o mais antigo (ou ignora — escolher).
- `onDraw(Canvas)`:
  - Desenha a imagem de fundo.
  - Para cada ponto: desenha círculo na cor do ponto + reta horizontal atravessando toda a largura (linha epipolar simulada).
- Método público `clearPoints()` para resetar.
- Método `getPoints(): List<EpipolarPoint>`.

### 🔹 Passo 7 — ViewModel

#### 7.1 `ModeloCameraViewModel.java`
- Estende `androidx.lifecycle.ViewModel`.
- Campos privados:
  - `MutableLiveData<CameraIntrinsics> intrinsics`
  - `MutableLiveData<DistortionCoefficients> distortion`
  - `MutableLiveData<Bitmap> originalImage`
  - `MutableLiveData<Bitmap> correctedImage`
  - `MutableLiveData<List<EpipolarPoint>> epipolarPoints`
  - `MutableLiveData<String> statusMessage`
- Expor cada um como `LiveData` (não `MutableLiveData`) via getter.
- Métodos:
  - `void loadDefaultCalibration()` — usa Repository + Factory.
  - `void onImageCaptured(Mat originalMat)` — converte para Bitmap, executa `ImageUndistortStrategy`, salva ambas em LiveData.
  - `void onPointSelected(Point screenCoord)` — adiciona à lista, executa `PointUndistortStrategy`, atualiza LiveData.
  - `void resetEpipolar()`
- **Não** instanciar Strategy direto — receber via construtor (dependency injection manual):
  ```java
  public ModeloCameraViewModel(UndistortStrategy<Mat,Mat> imgStrategy,
                                 UndistortStrategy<Point,Point> ptStrategy,
                                 CalibrationRepository repo) { ... }
  ```
  E usar `ViewModelProvider.Factory` customizado para criar:
  ```java
  public class ModeloCameraVMFactory implements ViewModelProvider.Factory {
      @Override public <T extends ViewModel> T create(Class<T> modelClass) {
          return (T) new ModeloCameraViewModel(
              new ImageUndistortStrategy(),
              new PointUndistortStrategy(),
              CalibrationRepository.getInstance());
      }
  }
  ```

### 🔹 Passo 8 — Activity (UI)

#### 8.1 `ModeloCameraActivity.java`
- Estende `AppCompatActivity`.
- `onCreate`:
  1. `setContentView(R.layout.mc_activity_modelo_camera)`.
  2. Inicializa OpenCV (chamar `OpenCVLoader.initDebug()` se ainda não foi feito — **idempotente**).
  3. Cria ViewModel via `new ViewModelProvider(this, new ModeloCameraVMFactory()).get(ModeloCameraViewModel.class)`.
  4. Faz binding dos LiveData → views.
  5. Conecta listeners dos botões.
- Layout `mc_activity_modelo_camera.xml`:
  - Topo: TextView mostrando K e D atuais (já existe no app — manter).
  - Botões: "📷 Capturar A4", "🖼 Carregar da galeria", "↩ Linhas epipolares", "💾 Salvar comparação", "📤 Compartilhar".
  - Centro: `ComparisonImageView` ocupando ~60% da tela.
  - Modo "Linhas epipolares" troca `ComparisonImageView` por `EpipolarOverlayView` (use `ViewSwitcher` ou `Fragment`).
- ActivityResult APIs:
  ```java
  ActivityResultLauncher<Intent> cameraLauncher = registerForActivityResult(...);
  ActivityResultLauncher<Intent> galleryLauncher = registerForActivityResult(...);
  ```
- Sempre chamar métodos do ViewModel — **nunca** fazer chamada OpenCV direto da Activity.

### 🔹 Passo 9 — Integração com MainActivity (não-destrutiva)

- Abrir o `activity_main.xml` existente.
- **Adicionar** um novo botão `<Button android:id="@+id/btnModeloCamera" android:text="Modelo de Câmera" />` em algum espaço livre. **Não remover** os botões existentes do PCC.
- Em `MainActivity.java`:
  ```java
  findViewById(R.id.btnModeloCamera).setOnClickListener(v ->
      startActivity(new Intent(this, ModeloCameraActivity.class)));
  ```
- Adicionar a Activity no `AndroidManifest.xml`:
  ```xml
  <activity android:name=".modelocamera.ModeloCameraActivity"
            android:label="Modelo de Câmera"
            android:parentActivityName=".MainActivity"
            android:exported="false" />
  ```

### 🔹 Passo 10 — Testes unitários (mínimo viável)

Criar em `app/src/test/java/com/example/visionproject/modelocamera/`:

#### 10.1 `CameraIntrinsicsTest.java`
- Testa que `toMat()` produz Mat 3×3 com valores corretos.
- Testa `getHorizontalFOVDegrees(1920)` retorna ~68° para fx=1408.
- Testa que construtor lança `IllegalArgumentException` para fx ≤ 0.

#### 10.2 `DistortionCoefficientsTest.java`
- Testa que `getDistortionType()` retorna `BARREL` para k1 = -0.3.
- Testa que retorna `PINCUSHION` para k1 = +0.12.
- Testa que retorna `NONE` para k1 = 0.

#### 10.3 `IntrinsicsFactoryTest.java`
- Testa que `defaultFor1080p()` cria intrinsics com cx ≈ 960 e cy ≈ 540.

**Strategies não precisam de teste unitário** — dependem de OpenCV nativo, teste manual via app.

---

## 4. Critérios de Aceite (DoD — Definition of Done)

A implementação só é considerada pronta quando **todos** estes critérios estão verificados:

- [ ] App compila sem warnings novos
- [ ] Botão "PCC" continua funcionando exatamente como antes (testar manualmente)
- [ ] Toggle do PCC ON/OFF continua funcionando
- [ ] Geração de CSV do PCC continua funcionando
- [ ] Novo botão "Modelo de Câmera" abre a `ModeloCameraActivity`
- [ ] Activity exibe corretamente K e D iniciais
- [ ] "Capturar A4" abre câmera, retorna foto, exibe original + corrigida lado a lado
- [ ] "Carregar da galeria" abre seletor, retorna foto, exibe original + corrigida lado a lado
- [ ] "Linhas epipolares" permite tocar até 5 pontos, desenha círculos coloridos + linhas horizontais
- [ ] "Salvar comparação" salva PNG no `Pictures/VisionProject/`
- [ ] "Compartilhar" abre intent de compartilhamento com a imagem corrigida
- [ ] Voltar para a tela inicial via back button funciona
- [ ] Rotação da tela não derruba o estado (ViewModel sobrevive, fotos continuam exibidas)
- [ ] Testes unitários (mínimo `CameraIntrinsicsTest` e `DistortionCoefficientsTest`) passam

---

## 5. Anti-padrões a Evitar (lista do que NÃO fazer)

- ❌ **Não** fazer chamada OpenCV direto na Activity (sempre via Strategy/ViewModel).
- ❌ **Não** usar `static` em variáveis de estado mutável (use Singleton thread-safe ou DI).
- ❌ **Não** colocar lógica de undistort dentro de `onClick` ou `Runnable` ad-hoc.
- ❌ **Não** ler/salvar arquivo no thread principal (`AsyncTask` ou `ExecutorService`).
- ❌ **Não** hardcodar caminhos de arquivo — sempre via MediaStore ou `Context.getExternalFilesDir()`.
- ❌ **Não** mexer em `PccModule.java`, mesmo "só pra refatorar".
- ❌ **Não** alterar versão do OpenCV ou do Gradle (manter o que já está).
- ❌ **Não** adicionar bibliotecas externas grandes (Glide, Picasso, etc.) — use `BitmapFactory` puro. Excessão: `PhotoView` para zoom, se necessário.
- ❌ **Não** acoplar `ModeloCameraActivity` com `PccActivity` (zero imports cruzados).

---

## 6. Material de referência interno

- Material da disciplina: `cap2_ativ3.pdf` (Módulo 2 — Formação da Imagem e Modelo de Câmera).
- Valores típicos para smartphone 1080p (do material):
  - K: fx=fy≈1400, cx≈960, cy≈540
  - D: k1 ∈ [-0.4, -0.2], k2 ∈ [0.1, 0.2], k3 ≈ 0, p1, p2 < 0.002
- Documentação OpenCV4Android — `Calib3d.undistort()`, `Calib3d.undistortPoints()`.

---

## 7. Ordem de execução resumida (cheat-sheet para a IA)

1. Criar pacote `modelocamera/` e subpacotes `model/`, `strategy/`, `factory/`, `repository/`, `ui/`.
2. Criar pacote `shared/` (se não existir).
3. Implementar **classes de domínio** (Passo 1): `CameraIntrinsics`, `DistortionCoefficients`, `EpipolarPoint`. **Escrever testes unitários junto.**
4. Implementar **Singleton** `CalibrationRepository` (Passo 2).
5. Implementar **Factory** `IntrinsicsFactory` (Passo 3). Testes.
6. Implementar **Strategies** `PointUndistortStrategy` e `ImageUndistortStrategy` (Passo 4).
7. Implementar **helpers** `ImageCaptureHelper` e `FileExporter` (Passo 5).
8. Implementar **custom views** `ComparisonImageView` e `EpipolarOverlayView` (Passo 6).
9. Implementar **ViewModel** `ModeloCameraViewModel` + `ModeloCameraVMFactory` (Passo 7).
10. Implementar **Activity** `ModeloCameraActivity` + layout XML (Passo 8).
11. **Integrar** com MainActivity adicionando o botão (Passo 9).
12. Adicionar **Manifest entry** para a nova Activity.
13. Rodar todos os testes unitários.
14. **Smoke test manual:** abrir app → testar PCC (deve funcionar igual) → abrir Modelo de Câmera → capturar A4 → ver comparação → testar epipolares → salvar → compartilhar.
15. Rodar checklist da seção 4.

---

## 8. O que NÃO está no escopo desta atividade

(Explicitar para evitar scope creep — se a outra IA quiser "ajudar" implementando algo daqui, recusar.)

- ❌ Calibração real da câmera (Módulo 3, atividade futura).
- ❌ Rastreamento de pontos / matching de features.
- ❌ Reconstrução 3D / SLAM.
- ❌ Realidade aumentada.
- ❌ Detecção de chessboard automática (apenas captura manual).
- ❌ Estéreo / múltiplas câmeras.

Tudo isso virá em módulos posteriores. **Não antecipar.**

---

**Autor da especificação:** Vinicius L. Alvarenga
**Data:** Maio/2026
**Versão:** 1.0
