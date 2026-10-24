#!/bin/bash
# 学習用データ整形: ステレオ→モノ、長い無音の短縮、ラウドネス正規化（2 パス）、長さ上限
# 使い方: prep_dataset.sh <入力wav> <出力dir> <名前> [最大秒数]
set -euo pipefail
IN="$1"; OUTDIR="$2"; NAME="$3"; MAXSEC="${4:-0}"
mkdir -p "$OUTDIR"
TMP="$OUTDIR/.${NAME}_stage1.wav"
OUT="$OUTDIR/${NAME}.wav"
# 1) モノ化 + 0.7 秒を超える無音を 0.4 秒に短縮（-45dB 未満を無音とみなす）
SR="silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.2:stop_periods=-1:stop_duration=0.7:stop_threshold=-45dB:stop_silence=0.4"
LIMIT=(-vn)
if [ "$MAXSEC" != "0" ]; then LIMIT=(-t "$MAXSEC"); fi
ffmpeg -hide_banner -loglevel error -y -i "$IN" -ac 1 -af "$SR" "${LIMIT[@]}" -ar 48000 -c:a pcm_s16le "$TMP"
# 2) loudnorm 2 パス（-20 LUFS, TP -2dB）
STATS=$(ffmpeg -hide_banner -nostats -i "$TMP" -af "loudnorm=I=-20:TP=-2:LRA=11:print_format=json" -f null - 2>&1 | sed -n '/^{/,/^}/p')
read -r I TP LRA TH OFF < <(printf '%s' "$STATS" | python3 -I -c 'import json,sys; d=json.load(sys.stdin); print(d["input_i"],d["input_tp"],d["input_lra"],d["input_thresh"],d["target_offset"])')
ffmpeg -hide_banner -loglevel error -y -i "$TMP" -af "loudnorm=I=-20:TP=-2:LRA=11:measured_I=$I:measured_TP=$TP:measured_LRA=$LRA:measured_thresh=$TH:offset=$OFF:linear=true:print_format=summary" -ar 48000 -c:a pcm_s16le "$OUT" 2>&1 | grep -E "Output Integrated|Output True Peak" || true
rm -f "$TMP"
afinfo "$OUT" | grep -E "estimated duration"
