//! 音声ファイルの読み書きとリサンプリング。

use anyhow::{anyhow, bail, Context, Result};
use std::path::Path;

use symphonia::core::audio::sample::Sample;
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::errors::Error as SymError;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

/// モノラル f32 波形とサンプルレート。
#[derive(Clone, Debug)]
pub struct Mono {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

impl Mono {
    pub fn duration_secs(&self) -> f64 {
        self.samples.len() as f64 / self.sample_rate as f64
    }
}

/// 対応する入力拡張子（判定の目安。実際は内容で判別する）。
pub const INPUT_EXTENSIONS: &[&str] = &[
    "wav", "mp3", "flac", "ogg", "m4a", "aac", "mp4", "aiff", "aif", "mkv", "webm",
];

/// 音声ファイルをデコードしてモノラル化する（複数チャンネルは平均）。
pub fn decode(path: &Path) -> Result<Mono> {
    let file =
        std::fs::File::open(path).with_context(|| format!("開けません: {}", path.display()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }
    let mut format = symphonia::default::get_probe()
        .probe(
            &hint,
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .with_context(|| format!("音声形式を判別できません: {}", path.display()))?;
    let track = format
        .default_track(TrackType::Audio)
        .ok_or_else(|| anyhow!("音声トラックがありません: {}", path.display()))?;
    let params = track
        .codec_params
        .as_ref()
        .and_then(|p| p.audio())
        .ok_or_else(|| anyhow!("音声コーデック情報がありません: {}", path.display()))?;
    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(params, &AudioDecoderOptions::default())
        .with_context(|| format!("デコーダを作成できません: {}", path.display()))?;
    let track_id = track.id;

    let mut sample_rate: u32 = 0;
    let mut channels: usize = 0;
    let mut mono: Vec<f32> = Vec::new();
    let mut interleaved: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(Some(p)) => p,
            Ok(None) => break,
            Err(SymError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(anyhow!("読み込みエラー: {e}")),
        };
        if packet.track_id != track_id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(buf) => {
                if sample_rate == 0 {
                    sample_rate = buf.spec().rate();
                    channels = buf.spec().channels().count().max(1);
                }
                interleaved.resize(buf.samples_interleaved(), f32::MID);
                buf.copy_to_slice_interleaved(&mut interleaved);
                if channels == 1 {
                    mono.extend_from_slice(&interleaved);
                } else {
                    let inv = 1.0 / channels as f32;
                    for frame in interleaved.chunks_exact(channels) {
                        mono.push(frame.iter().sum::<f32>() * inv);
                    }
                }
            }
            Err(SymError::DecodeError(_)) => continue,
            Err(e) => return Err(anyhow!("デコードエラー: {e}")),
        }
    }
    if sample_rate == 0 || mono.is_empty() {
        bail!("音声データを取り出せませんでした: {}", path.display());
    }
    Ok(Mono {
        samples: mono,
        sample_rate,
    })
}

/// 16bit PCM WAV として書き出す（±1 にクリップ）。
pub fn write_wav(path: &Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec)
        .with_context(|| format!("書き出せません: {}", path.display()))?;
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        w.write_sample(v)?;
    }
    w.finalize()?;
    Ok(())
}

/// 窓付き sinc 補間によるリサンプリング（モノラル）。
/// ダウンサンプリング時はカットオフを下げてエイリアシングを抑える。
pub fn resample(input: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || input.is_empty() {
        return input.to_vec();
    }
    let ratio = from as f64 / to as f64; // 出力 1 サンプルあたりの入力サンプル数
    let cutoff = if ratio > 1.0 { 0.95 / ratio } else { 0.95 };
    let half_taps = ((24.0 * ratio.max(1.0)).ceil() as isize).max(24);
    let beta = 8.0_f64;
    let out_len = ((input.len() as f64) / ratio).floor() as usize;
    let mut out = Vec::with_capacity(out_len);
    let n = input.len() as isize;
    for i in 0..out_len {
        let t = i as f64 * ratio;
        let center = t.floor() as isize;
        let mut acc = 0.0f64;
        let mut wsum = 0.0f64;
        for k in (center - half_taps + 1)..=(center + half_taps) {
            let d = t - k as f64;
            let x = d / half_taps as f64;
            if x.abs() >= 1.0 {
                continue;
            }
            let w = kaiser(x, beta) * sinc(cutoff * d) * cutoff;
            wsum += w;
            if k >= 0 && k < n {
                acc += w * input[k as usize] as f64;
            }
        }
        // 正規化（端点でも振幅を保つ）
        let v = if wsum.abs() > 1e-9 { acc / wsum } else { 0.0 };
        out.push(v as f32);
    }
    out
}

fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-12 {
        1.0
    } else {
        let px = std::f64::consts::PI * x;
        px.sin() / px
    }
}

/// Kaiser 窓（x ∈ [-1,1]）。
fn kaiser(x: f64, beta: f64) -> f64 {
    bessel_i0(beta * (1.0 - x * x).max(0.0).sqrt()) / bessel_i0(beta)
}

fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let y = x * x / 4.0;
    for k in 1..50 {
        term *= y / (k as f64 * k as f64);
        sum += term;
        if term < 1e-12 * sum {
            break;
        }
    }
    sum
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;

    #[test]
    fn resample_preserves_sine() {
        let from = 48000u32;
        let to = 16000u32;
        let f = 440.0f64;
        let input: Vec<f32> = (0..48000)
            .map(|i| (2.0 * std::f64::consts::PI * f * i as f64 / from as f64).sin() as f32)
            .collect();
        let out = resample(&input, from, to);
        assert_eq!(out.len(), 16000);
        // 端を除いて誤差を確認
        let mut max_err = 0.0f32;
        for i in 200..(out.len() - 200) {
            let expect = (2.0 * std::f64::consts::PI * f * i as f64 / to as f64).sin() as f32;
            max_err = max_err.max((out[i] - expect).abs());
        }
        assert!(max_err < 0.02, "max_err={max_err}");
    }
}
