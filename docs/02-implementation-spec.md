# 産業用外観異常検知システム 実装仕様書

## 1. 実装対象

Pythonを用いて、画像から正常・異常を判定し、異常箇所を可視化し、評価・誤判定分析・条件変更後の再評価まで行える産業用外観異常検知システムを実装する。

以下の2フェーズを同一の処理パイプラインで扱う。

### Phase 1

公開産業データセット **MVTec AD 2 / Vial** を利用して、システム全体の動作と評価処理を確認する。

### Phase 2

スマートフォンで撮影した**くるみの独自データセット**を使用する。

異常として以下を扱う。

- crack：割れ
- chip：欠け

色、模様、大きさ等の天然物としての個体差は原則正常とする。

---

# 2. 実装方針

以下を重視する。

1. Notebookへ処理ロジックを直接大量に記述しない。
2. Notebookは実験、操作、結果確認、可視化に使用する。
3. 実処理は再利用可能なPython Moduleへ分離する。
4. Phase 1とPhase 2で共通Pipelineを利用する。
5. Dataset固有処理だけAdapterとして切り替える。
6. 実験条件と結果をRun単位で保存する。
7. 同一条件を可能な範囲で再現できるようにする。
8. Datasetや大容量ModelをGitへ直接含めない。
9. 実測していない評価値をREADME等へ記載しない。
10. 不要なWeb UI、REST API、DB、Cloud環境等は追加しない。

---

# 3. 技術スタック

以下を使用する。

- Python 3
- Jupyter Notebook / JupyterLab
- OpenCV
- PyTorch
- Anomalib
- PatchCore
- NumPy
- pandas
- Matplotlib
- scikit-learn
- PyYAML
- pytest
- Ruff

Package管理は `pyproject.toml` を使用する。

Lock Fileを生成し、依存Versionを固定する。

Package Managerは `uv` を第一候補とする。

Versionについては実装時点で互換性のあるStable Versionを選択し、特に以下の互換性を確認すること。

- Python
- PyTorch
- torchvision
- anomalib
- OpenCV

---

# 4. Repository構成

以下を基本構成とする。

```text
industrial-visual-anomaly-detection/
│
├─ notebooks/
│  ├─ 01_phase1_mvtec_vial.ipynb
│  ├─ 02_phase2_walnut.ipynb
│  └─ 03_error_analysis.ipynb
│
├─ src/
│  └─ anomaly_detection/
│     ├─ __init__.py
│     │
│     ├─ datasets/
│     │  ├─ __init__.py
│     │  ├─ base.py
│     │  ├─ mvtec.py
│     │  └─ walnut.py
│     │
│     ├─ preprocessing/
│     │  ├─ __init__.py
│     │  └─ pipeline.py
│     │
│     ├─ models/
│     │  ├─ __init__.py
│     │  └─ patchcore.py
│     │
│     ├─ training/
│     │  ├─ __init__.py
│     │  └─ trainer.py
│     │
│     ├─ inference/
│     │  ├─ __init__.py
│     │  └─ predictor.py
│     │
│     ├─ evaluation/
│     │  ├─ __init__.py
│     │  ├─ metrics.py
│     │  ├─ threshold.py
│     │  └─ error_analysis.py
│     │
│     ├─ visualization/
│     │  ├─ __init__.py
│     │  └─ anomaly_visualizer.py
│     │
│     ├─ experiments/
│     │  ├─ __init__.py
│     │  └─ run_manager.py
│     │
│     └─ utils/
│        ├─ __init__.py
│        ├─ config.py
│        ├─ device.py
│        ├─ logging.py
│        └─ reproducibility.py
│
├─ configs/
│  ├─ phase1_mvtec_vial.yaml
│  └─ phase2_walnut.yaml
│
├─ data/
│  ├─ README.md
│  └─ .gitkeep
│
├─ runs/
│  └─ .gitkeep
│
├─ assets/
│
├─ tests/
│  ├─ test_config.py
│  ├─ test_preprocessing.py
│  ├─ test_threshold.py
│  ├─ test_metrics.py
│  └─ test_dataset_manifest.py
│
├─ pyproject.toml
├─ uv.lock
├─ .gitignore
└─ README.md
```

---

# 5. Configuration

システムの主要設定はYAMLで管理する。

Pythonコードに実験条件をHard Codeしない。

## Phase 1設定例

`configs/phase1_mvtec_vial.yaml`

```yaml
project:
  name: industrial-visual-anomaly-detection
  seed: 42

dataset:
  type: mvtec_ad2
  root: data/mvtec_ad2
  category: vial

model:
  name: patchcore

preprocessing:
  resize:
    enabled: true
    width: 256
    height: 256

  crop:
    enabled: false

  contrast:
    enabled: false

  clahe:
    enabled: false

  denoise:
    enabled: false

threshold:
  mode: auto
  manual_value: 0.5

runtime:
  device: auto

output:
  run_root: runs
  save_model: true
  save_predictions: true
  save_heatmaps: true
  save_overlays: true
```

## Phase 2設定例

`configs/phase2_walnut.yaml`

```yaml
project:
  name: industrial-visual-anomaly-detection
  seed: 42

dataset:
  type: walnut
  root: data/walnut
  manifest: data/walnut/manifest.csv

model:
  name: patchcore

preprocessing:
  resize:
    enabled: true
    width: 256
    height: 256

  crop:
    enabled: false

  contrast:
    enabled: false

  clahe:
    enabled: false

  denoise:
    enabled: false

threshold:
  mode: auto
  manual_value: 0.5

runtime:
  device: auto

output:
  run_root: runs
  save_model: true
  save_predictions: true
  save_heatmaps: true
  save_overlays: true
```

---

# 6. Dataset共通Interface

Dataset依存処理を共通Pipelineから分離する。

概念上、以下のInterfaceを提供する。

```python
class DatasetAdapter(Protocol):
    def prepare(self) -> None:
        ...

    def get_train_data(self):
        ...

    def get_validation_data(self):
        ...

    def get_test_data(self):
        ...

    def describe(self) -> dict:
        ...
```

実際にはAnomalibのAPI構造との相性を優先し、無理な独自抽象化は行わない。

目的はPhase 1 / Phase 2切替時に、

- Training
- Inference
- Evaluation

側を変更しなくてよい構成にすることである。

---

# 7. Phase 1：MVTec AD 2

## 7.1 Category

使用Categoryは以下に固定する。

```text
Vial
```

## 7.2 Dataset取得

MVTec AD 2本体はRepositoryへ含めない。

ライセンス条件をREADMEへ記載し、ユーザーが公式提供元から取得できる手順を記載する。

利用するAnomalib VersionがMVTec AD 2の自動取得またはDatamoduleを正式にサポートする場合は、その機能を優先して利用してよい。

ライセンス同意等により自動Downloadが不適切な場合は、手動配置方式とする。

## 7.3 Dataset検証

起動時に以下を検証する。

- Dataset Root存在
- Vial Category存在
- Training画像存在
- Test画像存在
- Ground Truth Mask利用可否

不足している場合は具体的な配置方法をError Messageとして表示する。

---

# 8. Phase 2：Walnut Dataset

## 8.1 Directory例

```text
data/walnut/
├─ raw/
│  ├─ good/
│  ├─ crack/
│  └─ chip/
│
├─ processed/
└─ manifest.csv
```

元画像は `raw/` に保存し、上書き禁止とする。

前処理済み画像をMaterializeする場合のみ `processed/` を使用する。

## 8.2 Manifest

以下のColumnを保持する。

```text
image_path
label
defect_type
split
object_id
```

例：

```csv
image_path,label,defect_type,split,object_id
raw/good/001.jpg,good,,train,W001
raw/good/002.jpg,good,,test,W002
raw/crack/101.jpg,defect,crack,test,W101
raw/chip/201.jpg,defect,chip,test,W201
```

`object_id` は同一個体を複数角度から撮影した場合のData Leakage防止に利用する。

同一 `object_id` をTrainとTestへ跨がせない。

1個体1画像の場合はFile単位Splitでよい。

## 8.3 Split

PatchCoreのTrainingには原則として正常画像のみを使用する。

異常画像はValidation/Testに使用する。

画像枚数およびSplit比率は固定値とせず、ConfigまたはDataset作成時に変更可能とする。

---

# 9. OpenCV Preprocessing

以下を独立したPipelineとして実装する。

- Resize
- Crop
- Contrast Adjustment
- CLAHE
- Denoise
- Normalization

全機能を必ず同時使用するわけではない。

YAMLによるEnable / Disableを可能とする。

概念Interface：

```python
class ImagePreprocessor:
    def __init__(self, config: PreprocessingConfig):
        ...

    def process(self, image: np.ndarray) -> np.ndarray:
        ...
```

処理順序を一箇所に定義し、Notebookから変更しない。

## 可視化

Notebookから、

```text
Original
Preprocessed
```

を横並びで確認できるFunctionを提供する。

---

# 10. PatchCore

Anomalibが提供するPatchCore実装を利用する。

PatchCore Algorithm自体を独自再実装しない。

以下をWrapperとして提供する。

- Model生成
- Training
- Model保存
- Model読み込み
- Inference

Anomalib Versionによって推奨APIが異なる場合は、そのVersionの公式APIへ従う。

---

# 11. Device Selection

以下を実装する。

```text
device=auto
```

の場合、

1. CUDA利用可能 → GPU
2. 利用不可 → CPU

とする。

利用DeviceをLogおよびRun Metadataへ保存する。

GPU必須にはしない。

---

# 12. Reproducibility

ConfigにRandom Seedを保持する。

以下へ可能な範囲で適用する。

- Python random
- NumPy
- PyTorch
- CUDA

完全なBit-level再現性までは保証しない。

実験時に以下を保存する。

- Seed
- Python Version
- PyTorch Version
- anomalib Version
- OpenCV Version
- Device
- Config Snapshot

---

# 13. Training

Training実行時は以下を行う。

1. Config読み込み
2. Device決定
3. Random Seed設定
4. Dataset読み込み
5. Dataset件数表示
6. Preprocessing設定読み込み
7. PatchCore生成
8. Training
9. Model Artifact保存
10. 実行情報保存
11. Log保存

Notebookでは上記を1～数Cellから実行できること。

---

# 14. Threshold

## 14.1 Auto Mode

初期状態では、

```yaml
threshold:
  mode: auto
```

とする。

使用中のAnomalib Versionに適切なAdaptive Threshold機能が存在する場合はそれを利用する。

利用できない場合はValidation Datasetに対してThreshold候補をSweepし、F1 Score等を利用して自動決定する。

## 14.2 Manual Mode

以下も利用可能とする。

```yaml
threshold:
  mode: manual
  manual_value: 0.65
```

Notebook上からThresholdを変更し、再評価できること。

Modelの再TrainingをせずThresholdのみ変更した場合は、その旨をRun Metadataへ記録する。

---

# 15. Inference

各画像について以下の結果を生成する。

```text
image_path
ground_truth
anomaly_score
threshold
prediction
```

Prediction：

```text
GOOD
NG
```

可能であればPixel-levelのAnomaly Mapも取得する。

結果をDataFrameとしてNotebook上に表示可能とする。

---

# 16. Visualization

以下を表示・保存する。

1. Original Image
2. Anomaly Heatmap
3. Original + Heatmap Overlay

実装が単純な場合は以下も追加する。

4. Contour
5. Bounding Box

ただしBounding Box等の追加によって本体実装を複雑化させない。

結果画像には最低限、

- Prediction
- Anomaly Score
- Threshold

を確認可能な形で表示する。

---

# 17. Image-level Metrics

以下を算出する。

- Accuracy
- Precision
- Recall
- F1 Score
- AUROC
- AUPR

さらに以下を表示する。

- Confusion Matrix
- True Positive
- True Negative
- False Positive
- False Negative

本システムではFalse Negativeを特に重要なErrorとして扱う。

---

# 18. Pixel-level Metrics

Ground Truth Maskを利用可能なDatasetのみ実行する。

Phase 1では可能な範囲で以下を算出する。

- Pixel AUROC
- AUPRO等、Anomalibで安定して利用可能なLocalization Metric

Phase 2のWalnut DatasetではMask作成を必須にしない。

したがってPhase 2ではImage-level評価のみでも正常動作とみなす。

---

# 19. Error Analysis

以下を抽出するFunctionを実装する。

```python
get_false_positives(...)
get_false_negatives(...)
```

Notebook上で誤判定を一覧表示する。

各画像について、

- Original
- Ground Truth
- Prediction
- Anomaly Score
- Heatmap

を確認する。

False Negativeを先に表示できるOptionを持たせる。

---

# 20. Experiment Run

Trainingまたは主要なEvaluationを実行するたびにRun Directoryを生成する。

例：

```text
runs/
└─ 20260926_163000_phase1/
   ├─ config.yaml
   ├─ metadata.json
   ├─ metrics.json
   ├─ predictions.csv
   ├─ run.log
   │
   ├─ model/
   │
   └─ images/
      ├─ heatmaps/
      └─ overlays/
```

Run IDはTimestampを基本とする。

---

# 21. Experiment Summary

各Run終了時にRepositoryの

```text
runs/summary.csv
```

へ結果を追記する。

最低限以下を保存する。

```text
run_id
phase
dataset
model
threshold_mode
threshold
preprocessing_summary
accuracy
precision
recall
f1
auroc
aupr
device
```

これにより改善前後を比較できるようにする。

例：

```text
run001
  Recall = 0.84

↓ CLAHEを有効化

run002
  Recall = 0.91
```

Notebookで複数Runを表または簡単なGraphとして比較できること。

---

# 22. 改善サイクル

システムとして以下のサイクルを実施可能とする。

```text
Training / Inference
        ↓
Evaluation
        ↓
False Positive / False Negative確認
        ↓
原因仮説
        ↓
条件変更
        ↓
再Trainingまたは再Evaluation
        ↓
Run比較
```

変更対象：

- Dataset
- Preprocessing
- Threshold
- Training Configuration

「改善機能」を自動化する必要はない。

人間が結果を分析し、Configを変更して再実行する方式とする。

---

# 23. Logging

標準 `logging` Module等を使用する。

ConsoleとFileへ出力する。

最低限以下を記録する。

- Execution Start
- Config Path
- Dataset
- Dataset件数
- Device
- Preprocessing Configuration
- Training Start/End
- Model保存
- Inference Start/End
- Threshold
- Metrics
- Run Directory
- Exception

---

# 24. Error Handling

以下について分かりやすいError Messageを出力する。

- Dataset不存在
- Dataset画像0件
- Config不存在
- Config値不正
- Model不存在
- Unsupported Image Format
- CUDA指定だがCUDA利用不可
- Manifest Column不足
- Trainに異常画像が混入している場合
- Train/Testのobject_id重複検出

Notebook上で巨大なStack Traceだけが表示される状態を避ける。

---

# 25. Test

pytestを使用する。

以下は必須。

## Config

- 正常YAML読込
- 必須項目不足
- 不正値

## Preprocessing

小さなSynthetic Imageを利用して、

- Resize
- CLAHE
- Denoise

等が期待Shapeで返ること。

## Threshold

Known Scoreに対して、

- GOOD
- NG

が正しく判定されること。

## Metrics

既知のPrediction/Ground Truthから期待する、

- Accuracy
- Precision
- Recall
- F1

が算出されること。

## Manifest

- 必須Column
- File existence
- Label validation
- object_id leakage

を検証する。

## Smoke Test

実データDownloadを必要としない小規模Synthetic Datasetで、主要ModuleのImportと最低限のInference経路が壊れていないことを確認する。

大規模PatchCore TrainingをCI/Testで実行しない。

---

# 26. Notebook 1

`01_phase1_mvtec_vial.ipynb`

以下のSectionを順番に配置する。

### 1. Environment

- Version表示
- Device表示

### 2. Config

Phase 1 Config読込。

### 3. Dataset Overview

- Dataset件数
- Sample正常画像
- Sample異常画像

### 4. Preprocessing

Original / Preprocessed比較。

### 5. Training

PatchCore Training実行。

### 6. Inference

Test Dataset推論。

### 7. Metrics

主要評価指標表示。

### 8. Visualization

Heatmap / Overlay表示。

### 9. Error Analysis

False Positive / False Negative確認。

### 10. Experiment Result

Run保存先およびSummary表示。

---

# 27. Notebook 2

`02_phase2_walnut.ipynb`

Phase 1と同じ基本構造とする。

追加で、

### Dataset Validation

- good件数
- crack件数
- chip件数
- Split件数
- object_id重複

を表示する。

Datasetが未配置の場合は、必要なDirectory構成とManifest作成方法を説明して終了する。

Notebookが壊れた状態にならないようにする。

---

# 28. Notebook 3

`03_error_analysis.ipynb`

複数Runの比較を目的とする。

以下を実装する。

- Run Summary読込
- Metrics比較Table
- Recall比較
- Precision比較
- F1比較
- Threshold比較
- False Positive表示
- False Negative表示

必要以上のDashboard化は行わない。

Matplotlibによる単純なChartでよい。

---

# 29. README

READMEを完成成果物として扱う。

以下のSectionを作成する。

## Overview

システム概要。

## Architecture

処理PipelineをDiagramまたはMermaidで表示。

```text
Dataset
→ OpenCV
→ PatchCore
→ Inference
→ Threshold
→ Visualization
→ Evaluation
→ Error Analysis
```

## Features

実装機能一覧。

## Tech Stack

使用技術。

## Project Structure

Directory説明。

## Setup

`uv` を使ったEnvironment構築方法。

## Dataset

### MVTec AD 2

Datasetの取得元・License・配置方法。

### Walnut Dataset

必要なDirectory構造とManifest形式。

## Run Phase 1

Notebook起動方法。

## Run Phase 2

同上。

## Evaluation Metrics

各Metricの簡潔な説明。

## Experiment Comparison

Run比較方法。

## Results

実際に実験した後にのみ記載する。

実行前に架空の精度を記載しない。

## License / Dataset Notice

Source Code LicenseとDataset Licenseを混同しない。

---

# 30. Git管理

`.gitignore` に最低限以下を設定する。

```text
.venv/
__pycache__/
.ipynb_checkpoints/

data/*
!data/README.md
!data/.gitkeep

runs/*
!runs/.gitkeep

*.ckpt
*.pt
*.pth
```

必要に応じてREADME用Sample画像のみ `assets/` へCommitする。

---

# 31. 実装しないもの

以下は実装しない。

- React等によるWeb UI
- Streamlit
- REST API
- Database
- Authentication
- AWS
- Docker必須化
- Kubernetes
- MLflow
- DVC
- Message Queue
- Production Model Serving
- Live Camera
- PLC連携
- C++推論Application

これらを「実務的だから」という理由だけで勝手に追加しない。

---

# 32. 実装優先順位

以下の順番で実装する。

## Step 1

Repository Skeleton / Package / Config / Logging / Device / Seed。

## Step 2

MVTec AD 2 Dataset Adapter。

## Step 3

OpenCV Preprocessing Pipeline。

## Step 4

PatchCore Wrapper。

## Step 5

Training / Model Artifact保存。

## Step 6

Inference。

## Step 7

Threshold / Image-level Metrics。

## Step 8

Heatmap / Overlay。

## Step 9

Error Analysis。

## Step 10

Run Manager / Experiment Summary。

## Step 11

Phase 1 Notebook。

## Step 12

Walnut Dataset Adapter / Manifest Validation。

## Step 13

Phase 2 Notebook。

## Step 14

Experiment Comparison Notebook。

## Step 15

pytest / Ruff。

## Step 16

README。

---

# 33. 実装時の判断ルール

仕様とLibrary APIが衝突した場合は、以下の優先順位で判断する。

1. 正しく動作すること
2. 使用Libraryの公式推奨API
3. 本仕様のArchitecture意図
4. コードの単純さ
5. 過剰な抽象化を避ける

AnomalibのVersion差によりClass名やAPIが変わっている場合は、現在使用するVersionへ合わせて実装してよい。

ただし、以下の機能要件は削除しない。

- OpenCV Preprocessing
- PatchCore
- Training
- Inference
- Threshold
- Heatmap
- Image-level Evaluation
- False Positive / False Negative分析
- Run単位の条件・結果保存
- Phase 1 / Phase 2共通Pipeline
- Notebookによる操作・分析

---

# 34. 完了条件

以下を満たせば実装完了とする。

## Code Quality

- `ruff check` が成功する
- `pytest` が成功する
- Notebook外の主要FunctionにType Hintがある
- Hard CodeされたAbsolute Pathがない

## Phase 1

- MVTec AD 2 / Vialを読み込める
- PatchCoreをTrainingできる
- Modelを保存できる
- Test Datasetを推論できる
- Anomaly Scoreを出力できる
- GOOD / NG判定できる
- Heatmapを生成できる
- Recall / Precision / F1 / Accuracy / AUROC / AUPRを確認できる
- False Positive / False Negativeを表示できる
- Run結果を保存できる

## Phase 2

- Walnut Datasetを読み込める
- ManifestをValidationできる
- Phase 1と同じPipelineでTrainingできる
- crack / chipを異常としてTestできる
- 評価・Heatmap・誤判定分析ができる
- 条件変更後のRunと結果比較できる

## Documentation

- READMEのみを読んでEnvironment構築方法が分かる
- Dataset配置方法が分かる
- Phase 1 / Phase 2実行方法が分かる
- Architectureが分かる
- Dataset本体をRepositoryに含めず利用できる

---

# 35. Claude Codeへの実装指示

本仕様に基づいてRepositoryを実装すること。

実装開始前に既存Repositoryが存在する場合は、現在のFile構成と内容を確認し、既存コードを不必要に破壊しないこと。

実装中に仕様上判断が必要な場合は、単に機能を削除するのではなく、本仕様の目的を維持した最も単純な実装を選択すること。

以下は禁止する。

- 架空の実験結果を生成する
- Datasetが存在しない状態で成功したように見せる
- 実行していないTestを成功したと記載する
- 使用していない技術をREADMEへ記載する
- 不要なWeb Application化
- 不要なCloud化
- 過剰なDesign Pattern導入
- Dataset Licenseを無視した再配布

実装後は以下を実際に実行して確認すること。

```text
dependency install
ruff check
pytest
```

可能な範囲でPhase 1の実Datasetを使用したEnd-to-End実行も行い、問題があれば修正すること。

最後に以下を報告すること。

1. 実装した機能
2. Repository構成
3. Test結果
4. Phase 1実行結果
5. 未実施項目
6. Dataset準備など利用者側で必要な作業
7. 既知の制約