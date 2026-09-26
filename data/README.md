# data/

Dataset本体はGit管理の対象外です。このディレクトリに各自で配置してください。
`data/README.md` と `data/.gitkeep` 以外は `.gitignore` で除外されています。

## MVTec AD 2 / Vial（Phase 1）

- 提供元: MVTec Software GmbH
- 取得方法: 公式ページ <https://www.mvtec.com/company/research/datasets/mvtec-ad-2> のフォームから取得する
- License: CC BY-NC-SA 4.0（商用利用不可）。Dataset本体および派生物を本Repositoryで再配布しない
- 使用するCategory: `vial` のみ

配置構造:

```text
data/mvtec_ad2/
└─ vial/
   ├─ train/good/*.png
   ├─ validation/good/*.png
   ├─ test_public/
   │  ├─ good/*.png
   │  ├─ bad/*.png
   │  └─ ground_truth/bad/*_mask.png
   ├─ test_private/          （配置してもよいが使用しない）
   └─ test_private_mixed/    （配置してもよいが使用しない）
```

- `train/good`: PatchCoreの学習（Memory Bank構築）に使用する
- `test_public`: 閾値決定用（validation）と最終評価用（test）へ層化分割して使用する。
  割合は `configs/phase1_mvtec_vial.yaml` の `dataset.threshold_split_ratio` で変更できる
- `validation/good`: 正常画像のみのためF1による閾値決定に使えず、初期Versionでは使用しない
- `test_private` / `test_private_mixed`: 正解ラベルが公開されていないため使用しない

## Walnut Dataset（Phase 2）

```text
data/walnut/
├─ raw/
│  ├─ good/    正常（明確な欠け・割れがない。色・大きさ・形状・模様の個体差は正常）
│  ├─ crack/   割れ
│  └─ chip/    欠け
└─ manifest.csv
```

元画像（`raw/`）はPipelineから読み込むだけで変更しません。前処理は読み込み時に適用するため、
`processed/` は使用しません。

### ファイル名とobject_id

- ファイル名は `<object_id>_<連番>.<拡張子>` とする（例: `G001_1.jpg`, `G001_2.jpg`）
- `object_id` はファイル名の最初の `_` より前の部分。`_` を含まない場合はファイル名（拡張子なし）全体
- 同じ個体を複数角度で撮影した画像は同じ `object_id` にする。同じ `object_id` の画像は
  必ず同じsplitに入り、Train / Testに跨がらない（Data Leakage防止）
- `object_id` はDataset全体で一意にする。good / crack / chip で同じ `object_id` を使うとエラーになる
  （例: 正常 `G001`、割れ `C001`、欠け `K001`）
- 1個体1画像の場合も同じ規則でよい（例: `G001_1.jpg`）
- 対応形式: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`。iPhoneのHEICはJPEGへ変換してから配置する
- スマートフォン画像のEXIF回転情報は読み込み時に反映される

### manifest.csv

列は `image_path,label,defect_type,split,object_id`（`image_path` は `data/walnut/` からの相対パス）。

```csv
image_path,label,defect_type,split,object_id
raw/good/G001_1.jpg,good,,train,G001
raw/good/G002_1.jpg,good,,test,G002
raw/crack/C001_1.jpg,defect,crack,test,C001
raw/chip/K001_1.jpg,defect,chip,validation,K001
```

- `label`: `good` / `defect`、`defect_type`: goodは空、defectは `crack` / `chip`
- `split`: `train`（学習、正常のみ） / `validation`（Threshold決定） / `test`（最終評価）

画像を配置した後、`notebooks/02_phase2_walnut.ipynb` の「Manifest作成」セルで `BUILD_MANIFEST = True`
にするか、次のコードで作成できます。split比率は `configs/phase2_walnut.yaml` の `dataset.split` で変更します
（既定: 正常 train 0.6 / validation 0.2 / test 0.2、異常 validation 0.5 / test 0.5。object_id単位で割り当て）。

```python
from anomaly_detection.datasets.walnut import WalnutAdapter
from anomaly_detection.utils.config import load_config

config = load_config("configs/phase2_walnut.yaml")
WalnutAdapter.from_config(config).build_manifest()  # 既存のmanifest.csvは overwrite=True の時だけ上書き
```

作成後にmanifest.csvを手で修正することもできます。読み込み時に、必須列・ファイル存在・値・
Trainへの異常画像混入・object_idのsplit跨ぎを検証します。
