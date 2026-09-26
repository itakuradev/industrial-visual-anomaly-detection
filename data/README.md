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
│  ├─ good/
│  ├─ crack/
│  └─ chip/
├─ processed/
└─ manifest.csv
```

撮影したファイルの命名規則とManifestの作成方法は、Phase 2の実装時に追記します。
