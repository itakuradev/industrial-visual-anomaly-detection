# 産業用外観異常検知システム PoC 要件定義書

## 1. 概要

### 1.1 システム名

Industrial Visual Anomaly Detection PoC

### 1.2 目的

製品画像を入力として、正常・異常を判定するとともに異常箇所を可視化し、モデルの評価および検出精度の改善を行える産業用外観異常検知システムを構築する。

システムは以下の一連の処理を対象とする。

**画像データ準備 → OpenCVによる画像前処理 → 異常検知モデル学習 → 推論 → 異常判定 → 異常箇所可視化 → 性能評価 → 誤判定分析 → 条件変更 → 再評価**

公開データセットと独自データセットの双方を同一の処理パイプラインで扱える構成とする。

---

# 2. 開発フェーズ

## 2.1 Phase 1：公開産業データセットによる検証

MVTec AD 2 の **Vial** データセットを使用する。

目的は、以下の処理が正常に機能することを確認することである。

- Dataset読み込み
- 画像前処理
- モデル学習
- モデル保存
- 推論
- 異常スコア算出
- 正常／異常判定
- 異常箇所可視化
- 評価指標算出
- 誤判定抽出
- 条件変更後の再評価

Phase 1ではMVTec AD 2側で提供されるデータ構成および正解情報を利用する。

---

## 2.2 Phase 2：独自データセットによる検証

対象物として**くるみ**を使用する。

スマートフォンで撮影した画像から独自Datasetを作成し、Phase 1と同じ異常検知Pipelineを使用する。

### 正常

以下を満たすものを正常とする。

- 明確な欠けがない
- 明確な割れがない

色、大きさ、形状、模様など天然物として通常発生する個体差は正常範囲とする。

### 異常

以下を異常として扱う。

- 欠け
- 割れ

Datasetの画像枚数は事前固定せず、検証結果を確認しながら必要量を決定する。

---

# 3. システム構成

主要技術として以下を使用する。

| 分類 | 技術 |
|---|---|
| 言語 | Python 3 |
| 実験・操作環境 | Jupyter Notebook |
| 画像処理 | OpenCV |
| Deep Learning基盤 | PyTorch |
| 異常検知Framework | Anomalib |
| 異常検知Model | PatchCore |
| 可視化 | Matplotlib |
| 評価 | scikit-learn / Anomalib Metrics |
| 設定 | YAML |
| テスト | pytest |

依存関係は `pyproject.toml` およびLock Fileによって固定する。

---

# 4. システムアーキテクチャ

処理Pipelineは以下とする。

```text
Dataset
   ↓
Dataset Loader
   ↓
OpenCV Preprocessing
   ↓
PatchCore
   ↓
Training
   ↓
Model Artifact
   ↓
Inference
   ↓
Anomaly Score / Anomaly Map
   ↓
Threshold
   ↓
OK / NG
   ↓
Visualization
   ↓
Metrics
   ↓
Error Analysis
   ↓
Experiment Comparison
```

Phase 1とPhase 2ではDataset Loaderのみ切り替え、後続Pipelineを共通利用する。

```text
                    ┌─ MVTec AD 2 / Vial
Dataset Adapter ────┤
                    └─ Custom Walnut Dataset
                             ↓
                      Common Pipeline
```

---

# 5. Dataset要件

## 5.1 Dataset Adapter

データ取得処理をDatasetごとに分離する。

以下の2種類を実装する。

- MVTec AD 2 Adapter
- Custom Dataset Adapter

モデル、評価、推論処理はDataset固有の構造に依存させない。

---

## 5.2 Dataset分割

Phase 2ではDatasetを以下の用途に分割可能な構成とする。

- Training
- Validation
- Test

PatchCoreのTrainingでは原則として正常画像を使用する。

Validation/Testでは正常画像と異常画像を使用し、判定性能を評価する。

Dataset枚数およびSplit比率は設定ファイルから変更可能とする。

---

## 5.3 元画像の保持

撮影・取得した元画像は変更しない。

```text
data/
├─ raw/
└─ processed/
```

のようにRaw DataとProcessed Dataを分離する。

OpenCVによる加工結果は元画像へ上書きしない。

---

## 5.4 Dataset Manifest

独自Datasetについて、画像情報をCSV等で管理可能とする。

最低限以下を保持する。

```text
image_path
label
split
defect_type
```

例：

```text
walnut_001.jpg,good,train,
walnut_002.jpg,good,test,
walnut_101.jpg,defect,test,crack
walnut_102.jpg,defect,test,chip
```

---

# 6. 画像前処理

OpenCVを使用して画像前処理を行う。

初期段階で処理を固定せず、必要に応じて以下を利用可能な構成とする。

- Resize
- Crop
- Contrast adjustment
- CLAHE
- Noise reduction
- Normalization

前処理は設定ファイルからON/OFFを変更できるものとする。

例：

```yaml
preprocessing:
  resize: true
  contrast: false
  clahe: false
  denoise: false
```

Notebook上で処理前・処理後の画像を比較表示できること。

---

# 7. 異常検知Model

## 7.1 Model

PatchCoreを使用する。

学習時には正常画像から特徴量を抽出し、正常画像の特徴分布を表現するMemory Bankを構築する。

本システムではModelの内部アルゴリズム自体を独自実装せず、Anomalib/PyTorchの実装を利用する。

---

## 7.2 学習

以下を実行可能とする。

- Dataset読み込み
- Training実行
- 学習結果表示
- Model Artifact保存
- 学習条件保存

同一条件で可能な範囲で再現できるようRandom Seedを設定する。

実行環境やライブラリ差による完全な一致までは保証対象としない。

---

# 8. 推論

任意のTest画像に対して推論を実施できること。

推論結果として最低限以下を取得する。

- 画像ファイル名
- Ground Truth
- Anomaly Score
- Threshold
- Prediction
- Anomaly Map

Predictionは以下とする。

```text
GOOD
NG
```

---

# 9. Threshold

異常スコアを正常／異常へ変換するためThresholdを使用する。

初期Thresholdは評価データから自動算出可能とする。

またNotebookまたは設定ファイルから手動変更可能とする。

例：

```yaml
threshold:
  mode: auto
  manual_value: 0.60
```

これによりThreshold変更による、

- 見逃し
- 過検知

の変化を確認できる構成とする。

---

# 10. 異常箇所可視化

推論結果について異常箇所を画像上で可視化する。

最低限以下を実装する。

- Original Image
- Anomaly Heatmap
- Overlay Image

実装負荷が小さい場合は以下も表示する。

- 異常領域Contour
- Bounding Box

可視化画像はファイルとして保存可能とする。

---

# 11. 評価

## 11.1 Image-level評価

以下を算出する。

- Recall
- Precision
- F1 Score
- Accuracy
- AUROC
- AUPR

外観検査用途では、異常品を正常品として判定するFalse Negativeを重要なエラーとして扱う。

そのため主要な確認指標としてRecallを重視する。

---

## 11.2 Pixel-level評価

Ground Truth Maskが利用可能なDatasetでは、異常箇所のLocalization性能についても評価する。

利用可能な場合に以下を出力する。

- Pixel-level AUROC
- AUPRO等のLocalization Metric

独自くるみDatasetについてGround Truth Maskを作成しない場合、Pixel-level評価は必須としない。

---

# 12. 誤判定分析

評価後、以下の画像を抽出可能とする。

### False Positive

正常だがNGと判定された画像。

### False Negative

異常だがGOODと判定された画像。

Notebook上で誤判定画像を一覧表示し、

- Original Image
- Anomaly Score
- Prediction
- Ground Truth
- Heatmap

を確認できること。

---

# 13. 改善・再評価

一度の学習結果を最終結果とはせず、条件変更後に再評価可能とする。

変更対象として以下を想定する。

- OpenCV前処理
- Dataset
- Training条件
- Threshold

処理例：

```text
Experiment v1
       ↓
Evaluation
       ↓
False Positive / False Negative Analysis
       ↓
Condition Change
       ↓
Experiment v2
       ↓
Evaluation
       ↓
Result Comparison
```

---

# 14. Experiment管理

複雑なExperiment Tracking基盤は使用せず、各実行結果をRun単位で保存する。

例：

```text
runs/
└─ 20260926_153000/
   ├─ config.yaml
   ├─ metrics.json
   ├─ predictions.csv
   ├─ run.log
   ├─ model/
   └─ images/
      ├─ heatmaps/
      └─ overlays/
```

各Runには使用した設定をSnapshotとして保存する。

これにより、

**どの条件で、どの結果が得られたか**

を後から追跡できること。

Runの主要結果はSummary CSVにも追加する。

例：

```text
run_id,preprocessing,threshold,recall,precision,f1,auroc
run001,none,0.61,0.84,0.91,0.87,0.93
run002,clahe,0.63,0.92,0.89,0.90,0.96
```

---

# 15. Artifact管理

以下を実行成果物として保存する。

- Model Artifact
- 実行Config
- Metrics
- Prediction結果
- Heatmap
- Overlay Image
- Experiment Summary
- Execution Log

大容量のModel ArtifactおよびDatasetについてはGit Repositoryへの直接Commitを必須としない。

---

# 16. 再現性

可能な範囲で同一条件の実験を再実行できるよう以下を保存する。

- Random Seed
- Config
- Python Version
- Package Version
- Model
- Dataset情報
- 実行Device
- 実験結果

Random Seedは設定ファイルから変更可能とする。

---

# 17. Logging

Python標準Logging等を使用し、以下を記録する。

- Dataset読み込み開始・完了
- Dataset件数
- Training開始・完了
- Model保存先
- Inference開始・完了
- Threshold
- Evaluation結果
- Error
- 使用Device

ConsoleとLog File双方へ出力可能とする。

---

# 18. Error Handling

最低限以下を検出し、原因を確認可能なエラーメッセージを出力する。

- Dataset Directoryが存在しない
- 画像が0件
- 対応していない画像形式
- Model Fileが存在しない
- Config不正
- CUDA利用不可時

GPUが使用できない場合はCPUへFallbackする。

---

# 19. Jupyter Notebook

NotebookはUIおよび分析環境として使用する。

処理ロジックそのものをNotebookへ集中させず、Python Moduleとして分離する。

Notebookから以下を操作可能とする。

- Dataset確認
- Sample画像表示
- 前処理確認
- Training
- Inference
- Evaluation
- Error Analysis
- Experiment Comparison

Notebookは以下を想定する。

```text
notebooks/
├─ 01_phase1_mvtec_vial.ipynb
├─ 02_phase2_walnut.ipynb
└─ 03_error_analysis.ipynb
```

---

# 20. Python Module構成

想定構成を以下とする。

```text
src/
├─ datasets/
│  ├─ mvtec.py
│  └─ walnut.py
│
├─ preprocessing/
│  └─ pipeline.py
│
├─ models/
│  └─ patchcore.py
│
├─ training/
│  └─ trainer.py
│
├─ inference/
│  └─ predictor.py
│
├─ evaluation/
│  ├─ metrics.py
│  └─ error_analysis.py
│
├─ visualization/
│  └─ anomaly_visualizer.py
│
└─ utils/
   ├─ config.py
   ├─ logging.py
   └─ reproducibility.py
```

---

# 21. Repository構成

```text
industrial-visual-anomaly-detection/
│
├─ notebooks/
│
├─ src/
│
├─ tests/
│
├─ configs/
│  └─ default.yaml
│
├─ data/
│  └─ README.md
│
├─ runs/
│
├─ assets/
│
├─ pyproject.toml
├─ lock file
├─ .gitignore
└─ README.md
```

`data/` および大容量の `runs/` 内容は原則Git管理対象外とする。

---

# 22. Dataset公開方針

MVTec AD 2 Dataset本体はRepositoryに含めない。

READMEに以下を記載する。

- Dataset名称
- Dataset提供元
- License
- 取得方法
- 配置Directory

独自Datasetについても大量画像はGit Repositoryへ直接登録せず、必要に応じてサンプル画像のみ `assets/` へ配置する。

---

# 23. 実行環境

CPU環境でも基本機能を実行可能とする。

CUDA対応GPUが利用可能な場合はGPUを自動利用する。

概念的には以下とする。

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

GPUの存在を必須要件とはしない。

---

# 24. Software Quality

## 24.1 Type Hint

主要なPython FunctionにはType Hintを付与する。

## 24.2 Formatter / Linter

Ruff等を利用してCode FormatとLintを統一する。

## 24.3 Test

pytestを使用して最低限以下をTestする。

- Config読み込み
- Dataset Loader
- Image Preprocessing
- Threshold判定
- Metric計算
- 小規模なInference Smoke Test

Model全体のTrainingをUnit Testでは実行しない。

---

# 25. README

READMEには以下を記載する。

1. システム概要
2. Architecture
3. 使用技術
4. Setup方法
5. Dataset準備方法
6. Phase 1実行方法
7. Phase 2実行方法
8. Training方法
9. Inference方法
10. Evaluation方法
11. Experiment結果
12. Sample Heatmap
13. Directory構成
14. Dataset Licenseに関する注意事項

---

# 26. 完成条件

以下をすべて満たした時点で初期Versionを完成とする。

### Phase 1

- MVTec AD 2 / Vialを読み込める
- PatchCoreによる学習を実行できる
- 学習済みModelを保存できる
- Test画像に推論できる
- Anomaly Scoreを取得できる
- GOOD / NGを判定できる
- Heatmapを表示・保存できる
- Recall / Precision / F1 / Accuracyを確認できる
- AUROC / AUPRを確認できる
- False Positive / False Negativeを確認できる

### Phase 2

- 独自くるみDatasetを読み込める
- 正常画像からModelを構築できる
- 欠け・割れ画像を異常として評価できる
- Phase 1と同じPipelineを利用できる
- 誤判定画像を確認できる
- 条件変更後に再評価できる
- Experiment間の結果を比較できる

### Repository

- Setup手順がREADMEから再現できる
- Configによって主要条件を変更できる
- 実験条件と結果がRun単位で保存される
- Dataset本体に依存せずRepositoryを公開できる
- 基本的な自動Testが成功する

---

# 27. 対象外

初期Versionでは以下を対象外とする。

- Web Application UI
- REST API
- Database
- User Authentication
- Cloud Deployment
- AWS等のInfrastructure
- Mobile Application
- Docker必須化
- MLflow
- DVC
- Kubernetes
- Model Serving API
- リアルタイムカメラ入力
- PLC / FA機器連携
- C++による推論実装

これらは画像異常検知Pipelineそのものとは独立した拡張機能として扱う。

---

# 28. 将来拡張候補

必要に応じて以下を追加可能な構成とする。

- 複数異常検知Model比較
- PatchCore以外のModel
- Data Augmentation
- Camera Live Input
- ONNX Export
- OpenVINOによる推論
- C++推論Application
- 推論速度Benchmark
- Memory使用量計測
- MLflowによるExperiment Tracking
- DVCによるDataset Version管理
- API化
- Production Deployment

初期Versionではこれらを必須としない。