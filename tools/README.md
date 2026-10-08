# 開発用ツール（任意・Python）

アプリ本体は Python を使いません。ここにあるのは出力の検証用スクリプトです。

```sh
# 入力/出力のスペクトログラム・F0 中央値・ピーク/RMS を比較し spectrograms.png を出力
uv run --with numpy --with soundfile --with librosa --with matplotlib python -I tools/analyze_outputs.py
```

`analyze_outputs.py` 冒頭の `pairs` に (名前, 入力パス, 出力パス) を並べて使います。
