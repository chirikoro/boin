# boin — 音声ファイルを「愛想良い系少女の声 V2」に変換するアプリ

音声ファイル（wav / mp3 / flac / ogg / m4a など）を、RVC モデル「愛想良い系少女の声セット V2」の 5 種の声に変換します。
Rust + ONNX Runtime で動作し、**Python や PyTorch のインストールは不要**です。

| 声モデル | ファイル |
|---|---|
| さらさら風味ちゃん（既定） | `V2-AISO-SARASARA` |
| ほわっと風味ちゃん | `V2-AISO-HOWATTO` |
| かっこいい風味ちゃん | `V2-AISO-KAKKOII` |
| しっとり風味ちゃん | `V2-AISO-SITTORI` |
| しゃきっと風味ちゃん | `V2-AISO-SYAKITTO` |

5 モデルはいずれも **ピッチ無し（F0 なし）** の RVC v2 / 40kHz モデルです。声の高さはモデルが決めるため、ピッチ指定はありません。

## フォルダ構成

```
boin/
├── boin-gui(.exe)   # GUI
├── boin(.exe)       # CLI
├── models/          # V2-AISO-*.pth を置く（初回に同名の .onnx を自動生成）
└── assets/          # 基盤モデル（初回に自動ダウンロード）
```

## 使い方（Windows）

1. zip を展開し、`models/` に `V2-AISO-*.pth`（5 ファイル）をコピーします。
2. `boin-gui.exe` を起動します。初回は「モデルを準備」を押すと
   - ContentVec（特徴抽出モデル、約 378MB）を Hugging Face からダウンロードし SHA-256 を検証
   - `.pth` を純 Rust で `.onnx` に変換（1 モデル 2〜3 秒）
   
   を行います。2 回目以降は不要です。
3. 音声ファイルをウィンドウにドロップ（またはボタンで追加）し、声モデルを選んで「変換開始」。
4. 変換結果は `元ファイル名_モデル名.wav`（16bit / 40kHz）として、既定では入力と同じフォルダに保存されます。

インターネット接続が無い PC で使う場合は、別の PC で `assets/contentvec_768l12.onnx` と `models/*.onnx` を作って一緒にコピーしてください。

### 処理デバイス

- **CPU**（既定）: どの PC でも動きます。目安は実時間の 0.2〜0.5 倍（1 分の音声が 15〜30 秒）。
- **GPU (DirectML)**: Windows で DirectX 12 対応 GPU がある場合に選べます。登録に失敗した場合は自動的に CPU で動きます。
- **GPU (CoreML)**（macOS）: 手元の M4 Pro では CPU より遅かったため、Mac でも CPU 推奨です。

## CLI

```powershell
boin.exe setup                       # 初回準備（基盤モデル DL + 5 モデルの ONNX 変換）
boin.exe convert input.wav           # 既定: さらさら、入力と同じ場所に input_sarasara.wav
boin.exe convert *.mp3 -m howatto -o out\   # 一括変換、出力先指定
boin.exe convert in.wav --device directml --output-sr 48000 --overwrite
boin.exe models                      # 準備状況
boin.exe doctor                      # 環境診断（ONNX Runtime、デバイス、SHA-256）
boin.exe inspect models\V2-AISO-SARASARA.onnx   # ONNX の入出力表示
```

主なオプション: `-m/--model`（sarasara/howatto/kakkoii/sittori/syakitto）、`-o/--out-dir`、`--device cpu|directml|coreml`、`--output-sr`、`--rms-mix-rate 0〜1`（入力の音量エンベロープを混ぜる）、`--lite`（int8 量子化版 ContentVec、約 95MB）、`--threads`、`--seed`。

## ビルド（開発者向け）

Rust ツールチェーンは `rust-toolchain.toml` で固定しています（rustup が自動で取得）。

```sh
cargo build --release            # target/release/boin, boin-gui
cargo test --workspace
```

Windows 向けの配布 zip は GitHub Actions（`.github/workflows/build.yml`）が `windows-latest` で作成します。Windows PC 上で `cargo build --release` しても同じものが作れます（Visual Studio Build Tools の C++ ワークロードが必要）。

## 仕組み

1. 入力をデコードし 16kHz モノラルへリサンプル、48Hz ハイパス（RVC 公式と同じ）
2. 長い音声は無音に近い位置で約 40 秒ごとに分割（前後 1 秒の文脈付き）
3. ContentVec（ONNX）で 768 次元の内容特徴量を抽出
4. `.pth` から生成した生成器（ONNX）で 40kHz の音声を合成
5. 区間を結合し、必要なら指定サンプルレートへ変換して WAV 出力

ピッチ付き（F0 あり）の RVC v2 モデルを `models/` に置いた場合は、RMVPE（約 362MB）を自動取得してピッチ推定も行います。

## ライセンス・クレジット

- 本アプリ: MIT License
- `crates/vc-convert`: [shirohata/vc-rs](https://github.com/shirohata/vc-rs) v0.5.3 の `.pth → .onnx` 変換器（MIT）をベンダリングし、ピッチ無しモデルを受け付けるよう変更
- ContentVec ONNX: [TigreGotico/voiceclonnx-rvc](https://huggingface.co/TigreGotico/voiceclonnx-rvc)（MIT）
- RMVPE ONNX: [lj1995/VoiceConversionWebUI](https://huggingface.co/lj1995/VoiceConversionWebUI)（MIT）
- GUI フォント: BIZ UDPGothic（SIL Open Font License 1.1、`OFL-BIZUDPGothic.txt`）
- 声モデル「愛想良い系少女の声セット V2」は配布元の利用条件に従ってください（本リポジトリには含みません）
