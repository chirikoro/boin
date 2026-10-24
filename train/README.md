# 自前モデルの学習（Mac / Apple Silicon）

2 人の女性の声から「どちらでもない架空の声」の RVC モデルを作り、boin で使うまでの手順。
学習だけは Python（[Applio](https://github.com/IAHispano/Applio)）を使う。boin 本体の実行には引き続き Python は不要。

## 必要なもの

- Apple Silicon Mac（M4 Pro 24GB で動作確認。GPU=MPS を使う）
- `uv`, `ffmpeg`（Homebrew）
- 学習用音声: 話者ごとに WAV（モノ/ステレオ可、48kHz 推奨、BGM・残響なし）。1 話者 8〜15 分あれば十分。
- 空きディスク 15GB 以上（Python 環境 9GB ＋ 1 モデルあたり約 2GB）

## 1. Applio の準備（初回のみ）

```sh
cd train
git clone --depth 1 https://github.com/IAHispano/Applio.git
cd Applio
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
git apply ../applio-local.patch          # MPS で特徴抽出が落ちる不具合の修正
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python core.py prerequisites --pretraineds-hifigan --models --no-exe
```

`prerequisites` で事前学習モデル（f0G40k/f0D40k）、RMVPE、ContentVec が `rvc/models/` に入る（約 1.9GB）。

## 2. データ整形

```sh
./train/prep_dataset.sh <入力.wav> train/datasets/<名前> <名前> [最大秒数]
# 例
./train/prep_dataset.sh 女性の声.wav  train/datasets/fem_a fem_a 900   # 15 分に制限
./train/prep_dataset.sh 女性の声2.wav train/datasets/fem_b fem_b 0     # 全部使う
```

モノ化、0.7 秒を超える無音の短縮、ラウドネス正規化（-20 LUFS）を行う。

## 3. 前処理 → 特徴抽出 → 学習

```sh
./train/run_stage.sh preprocess fem_a train/datasets/fem_a   # 40kHz 化と区間分割（数十秒）
./train/run_stage.sh extract    fem_a                         # RMVPE ピッチ + ContentVec（1 分程度）
./train/run_stage.sh train      fem_a 300 8 25                # 300 エポック, batch 8, 25 エポックごとに保存
```

- 設定は RVC v2 / 40kHz / F0 あり（RMVPE）/ HiFi-GAN。boin のパイプラインにそのまま載る構成。
- 所要時間の目安（M4 Pro, MPS）: 15 分のデータで 1 エポック約 60 秒 → 300 エポックで約 5 時間。
- 出力は `train/Applio/logs/<名前>/` に入る。`<名前>_<epoch>e_<step>s.pth` が推論用の重み（約 55MB）。
  `G_*.pth` / `D_*.pth` は再開用の大きなチェックポイント（最新のみ保持）。
- 進捗は `train/train.log` か、`cd train/Applio && .venv/bin/python core.py tensorboard` で確認できる。
- 途中で止めても、同じコマンドを再実行すれば最後の保存地点から再開する。

## 4. 2 つのモデルを混ぜて架空の声にする

```sh
./train/run_stage.sh blend fem_blend \
  train/Applio/logs/fem_a/fem_a_300e_14700s.pth \
  train/Applio/logs/fem_b/fem_b_300e_9000s.pth 0.5
```

比率は **1 つ目のモデルの重み**（0.0〜1.0、0.1 刻み）。0.7 なら話者 1 が 7 割、話者 2 が 3 割。
結果は `train/Applio/logs/<新モデル名>.pth`。
気に入る比率が見つかるまで何度でも作り直せる（学習は不要、数秒で終わる）。
単体の話者も `train/Applio/logs/fem_a/fem_a_300e_14700s.pth` をそのまま `models/fem_a.pth` に
コピーすれば使える。

## 5. boin で使う

`.pth` を `models/` にコピーするだけ。GUI の声モデル一覧（「再読込」ボタン）と `boin models` に
ファイル名で現れ、初回選択時に自動で ONNX 化される。

```sh
cp train/Applio/logs/fem_blend.pth models/
cargo run --release -p boin-cli -- convert 入力.wav -m fem_blend -p 0
```

F0 ありモデルなので `-p`（半音）でピッチを上げ下げできる（RMVPE は初回に自動ダウンロード）。

## 注意

- 学習に使う声は、本人から「音声変換モデルの作成と業務利用」への同意を得ること。
- `train/Applio`, `train/datasets`, `*.wav` は git 管理外。
