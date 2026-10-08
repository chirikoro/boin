//! 変換パイプライン本体。RVC WebUI の非リアルタイム推論（infer/vc/pipeline.py）を再現する。

use crate::audio::{self, Mono};
use crate::dsp::{highpass_filtfilt, LogMel};
use crate::f0;
use crate::models::Voice;
use crate::onnx::{build_session, Device};
use crate::paths::Home;
use crate::prepare;
use anyhow::{anyhow, bail, Result};
use ort::session::Session;
use ort::value::TensorRef;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;
use std::path::Path;
use std::time::{Duration, Instant};

const SR: usize = 16_000;
const WINDOW: usize = 160;
const X_PAD: usize = 1;
const X_QUERY: usize = 6;
const X_CENTER: usize = 38;
const X_MAX: usize = 41;
const LATENT_CHANNELS: usize = 192;

/// 変換パラメータ。
#[derive(Clone, Debug)]
pub struct ConvertOptions {
    pub voice: Voice,
    /// ピッチ変更（半音）。+12 で 1 オクターブ上。
    pub pitch_semitones: f32,
    /// 出力サンプルレート（None ならモデル固有の 40000Hz）。
    pub output_sample_rate: Option<u32>,
    /// 音量エンベロープの混合率（1.0 = 変換後の音量をそのまま使う）。
    pub rms_mix_rate: f32,
    /// RMVPE の有声判定しきい値。
    pub f0_threshold: f32,
    /// 乱数シード（None なら毎回ランダム）。
    pub seed: Option<u64>,
}

impl Default for ConvertOptions {
    fn default() -> Self {
        ConvertOptions {
            voice: Voice::default(),
            pitch_semitones: 0.0,
            output_sample_rate: None,
            rms_mix_rate: 1.0,
            f0_threshold: f0::RMVPE_THRESHOLD,
            seed: None,
        }
    }
}

/// 進捗段階。
#[derive(Clone, Debug)]
pub enum Stage {
    LoadModel(Voice),
    Decode,
    Pitch,
    Segment { index: usize, total: usize },
    Write,
}

#[derive(Clone, Debug)]
pub struct ConvertReport {
    pub input_secs: f64,
    pub elapsed: Duration,
    pub output_sample_rate: u32,
    pub segments: usize,
    pub device: Device,
}

impl ConvertReport {
    /// 実時間比（1.0 未満なら実時間より速い）。
    pub fn realtime_factor(&self) -> f64 {
        self.elapsed.as_secs_f64() / self.input_secs.max(1e-9)
    }
}

struct Generator {
    voice: Voice,
    session: Session,
    sample_rate: u32,
    /// 生成器が pitch/pitchf 入力を持つ（F0 ありモデル）か。
    needs_f0: bool,
}

/// 変換器。基盤モデルを 1 度読み込んで使い回す。
pub struct Converter {
    home: Home,
    device: Device,
    threads: usize,
    lite: bool,
    embedder: Session,
    /// ContentVec が attention_mask 入力を持つか。
    embedder_has_mask: bool,
    /// F0 ありモデルの時だけ読み込む。
    rmvpe: Option<Session>,
    logmel: LogMel,
    generator: Option<Generator>,
}

impl Converter {
    /// 基盤モデル（ContentVec / RMVPE）を読み込む。事前に `prepare::ensure_base_models` が必要。
    pub fn new(home: &Home, device: Device, lite: bool, threads: usize) -> Result<Converter> {
        let emb_path = prepare::embedder_model(lite).path(home);
        if !emb_path.is_file() {
            bail!("基盤モデルが未準備です。`boin setup` または GUI の「モデルを準備」を実行してください。");
        }
        let embedder = build_session(&emb_path, device, threads)?;
        validate_embedder(&embedder)?;
        let embedder_has_mask = embedder
            .inputs()
            .iter()
            .any(|o| o.name() == "attention_mask");
        Ok(Converter {
            home: home.clone(),
            device,
            threads,
            lite,
            embedder,
            embedder_has_mask,
            rmvpe: None,
            logmel: LogMel::rmvpe(),
            generator: None,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn lite(&self) -> bool {
        self.lite
    }

    /// 声モデルを読み込む（必要なら .pth → .onnx 変換を行う）。
    pub fn load_voice(
        &mut self,
        voice: Voice,
        progress: &mut dyn FnMut(prepare::Progress),
    ) -> Result<()> {
        if self.generator.as_ref().map(|g| g.voice) == Some(voice) {
            return Ok(());
        }
        let onnx = prepare::ensure_voice_onnx(&self.home, voice, progress)?;
        let session = build_session(&onnx, self.device, self.threads)?;
        let needs_f0 = validate_generator(&session)?;
        if needs_f0 && self.rmvpe.is_none() {
            let rmvpe_path = prepare::RMVPE.path(&self.home);
            if !rmvpe_path.is_file() {
                prepare::download_verified(&prepare::RMVPE, &rmvpe_path, progress)?;
            }
            let rmvpe = build_session(&rmvpe_path, self.device, self.threads)?;
            validate_rmvpe(&rmvpe)?;
            self.rmvpe = Some(rmvpe);
        }
        let sample_rate = generator_sample_rate(&session).unwrap_or(40_000);
        self.generator = Some(Generator {
            voice,
            session,
            sample_rate,
            needs_f0,
        });
        Ok(())
    }

    /// 現在の声モデルがピッチ（F0）を使うか。
    pub fn current_voice_uses_f0(&self) -> Option<bool> {
        self.generator.as_ref().map(|g| g.needs_f0)
    }

    /// 現在読み込んでいる声モデル。
    pub fn current_voice(&self) -> Option<Voice> {
        self.generator.as_ref().map(|g| g.voice)
    }

    /// ファイルを変換して WAV に書き出す。
    pub fn convert_file(
        &mut self,
        input: &Path,
        output: &Path,
        opts: &ConvertOptions,
        progress: &mut dyn FnMut(Stage),
    ) -> Result<ConvertReport> {
        let started = Instant::now();
        progress(Stage::LoadModel(opts.voice));
        self.load_voice(opts.voice, &mut |_| {})?;
        progress(Stage::Decode);
        let mono = audio::decode(input)?;
        let (samples, sr, segments) = self.convert_samples(&mono, opts, progress)?;
        progress(Stage::Write);
        audio::write_wav(output, &samples, sr)?;
        Ok(ConvertReport {
            input_secs: mono.duration_secs(),
            elapsed: started.elapsed(),
            output_sample_rate: sr,
            segments,
            device: self.device,
        })
    }

    /// メモリ上の音声を変換する。戻り値は (波形, サンプルレート, 区間数)。
    pub fn convert_samples(
        &mut self,
        mono: &Mono,
        opts: &ConvertOptions,
        progress: &mut dyn FnMut(Stage),
    ) -> Result<(Vec<f32>, u32, usize)> {
        self.load_voice(opts.voice, &mut |_| {})?;
        let tgt_sr = self
            .generator
            .as_ref()
            .map(|g| g.sample_rate)
            .unwrap_or(40_000) as usize;

        // 16kHz モノラル + ハイパス
        let audio16 = audio::resample(&mono.samples, mono.sample_rate, SR as u32);
        if audio16.len() < WINDOW * 4 {
            bail!("音声が短すぎます（{} サンプル）", audio16.len());
        }
        let audio = highpass_filtfilt(&audio16);

        let t_pad = SR * X_PAD;
        let t_pad2 = t_pad * 2;
        let t_query = SR * X_QUERY;
        let t_center = SR * X_CENTER;
        let t_max = SR * X_MAX;
        let t_pad_tgt = tgt_sr * X_PAD;

        // 無音に近い位置で分割点を決める（長い音声のみ）
        let mut opt_ts: Vec<usize> = Vec::new();
        if audio.len() > t_max {
            let apad = reflect_pad(&audio, WINDOW / 2, WINDOW / 2);
            let mut prefix = vec![0.0f64; apad.len() + 1];
            for (i, v) in apad.iter().enumerate() {
                prefix[i + 1] = prefix[i] + v.abs() as f64;
            }
            let audio_sum = |i: usize| prefix[i + WINDOW] - prefix[i];
            let mut t = t_center;
            while t < audio.len() {
                let lo = t - t_query;
                let hi = (t + t_query).min(audio.len());
                let mut best = lo;
                let mut best_v = f64::MAX;
                for i in lo..hi {
                    let v = audio_sum(i);
                    if v < best_v {
                        best_v = v;
                        best = i;
                    }
                }
                opt_ts.push(best);
                t += t_center;
            }
        }

        // 前後 1 秒の反射パディング → 全体の F0
        let audio_pad = reflect_pad(&audio, t_pad, t_pad);
        let p_len = audio_pad.len() / WINDOW;
        let needs_f0 = self.generator.as_ref().map(|g| g.needs_f0).unwrap_or(false);
        let (pitch, pitchf) = if needs_f0 {
            progress(Stage::Pitch);
            self.get_f0(&audio_pad, p_len, opts)?
        } else {
            (vec![1i64; p_len], vec![0.0f32; p_len])
        };

        let total_segments = opt_ts.len() + 1;
        let mut rng: rand::rngs::StdRng = match opts.seed {
            Some(s) => rand::rngs::StdRng::seed_from_u64(s),
            None => rand::rngs::StdRng::from_os_rng(),
        };
        let mut out: Vec<f32> = Vec::with_capacity(audio.len() * tgt_sr / SR + 1);
        let mut s = 0usize;
        let mut last_t: Option<usize> = None;
        for (i, &t_raw) in opt_ts.iter().enumerate() {
            progress(Stage::Segment {
                index: i + 1,
                total: total_segments,
            });
            let t = t_raw / WINDOW * WINDOW;
            let end = (t + t_pad2 + WINDOW).min(audio_pad.len());
            let f_end = ((t + t_pad2) / WINDOW).min(p_len);
            let y = self.vc(
                &audio_pad[s..end],
                &pitch[s / WINDOW..f_end],
                &pitchf[s / WINDOW..f_end],
                &mut rng,
            )?;
            push_trimmed(&mut out, &y, t_pad_tgt);
            s = t;
            last_t = Some(t);
        }
        progress(Stage::Segment {
            index: total_segments,
            total: total_segments,
        });
        let start = last_t.unwrap_or(0);
        let y = self.vc(
            &audio_pad[start..],
            &pitch[start / WINDOW..],
            &pitchf[start / WINDOW..],
            &mut rng,
        )?;
        push_trimmed(&mut out, &y, t_pad_tgt);

        if (opts.rms_mix_rate - 1.0).abs() > 1e-6 {
            change_rms(
                &audio16,
                SR as u32,
                &mut out,
                tgt_sr as u32,
                opts.rms_mix_rate,
            );
        }

        let mut out_sr = tgt_sr as u32;
        if let Some(rs) = opts.output_sample_rate {
            if rs >= 16_000 && rs != out_sr {
                out = audio::resample(&out, out_sr, rs);
                out_sr = rs;
            }
        }
        // 公式と同じ正規化（最大振幅が 0.99 を超える場合のみ縮小）
        let peak = out.iter().fold(0.0f32, |a, &v| a.max(v.abs())) / 0.99;
        if peak > 1.0 {
            for v in out.iter_mut() {
                *v /= peak;
            }
        }
        Ok((out, out_sr, total_segments))
    }

    /// RMVPE で F0 を推定し、ピッチシフトと粗い量子化を行う。長さは p_len に揃える。
    fn get_f0(
        &mut self,
        audio_pad: &[f32],
        p_len: usize,
        opts: &ConvertOptions,
    ) -> Result<(Vec<i64>, Vec<f32>)> {
        let (mel, frames) = self.logmel.compute(audio_pad);
        let n_mels = self.logmel.n_mels();
        // フレーム数を 32 の倍数にゼロ埋め
        let n_pad = 32 * ((frames - 1) / 32 + 1) - frames;
        let padded = frames + n_pad;
        let mut mel_in = vec![0.0f32; n_mels * padded];
        for m in 0..n_mels {
            mel_in[m * padded..m * padded + frames]
                .copy_from_slice(&mel[m * frames..(m + 1) * frames]);
        }
        let rmvpe = self
            .rmvpe
            .as_mut()
            .ok_or_else(|| anyhow!("RMVPE が未読み込みです"))?;
        let input_name = rmvpe.inputs()[0].name().to_string();
        let output_name = rmvpe.outputs()[0].name().to_string();
        let tensor = TensorRef::from_array_view(([1usize, n_mels, padded], mel_in.as_slice()))?;
        let outputs = rmvpe.run(ort::inputs![input_name.as_str() => tensor])?;
        let (shape, data) = outputs[output_name.as_str()].try_extract_tensor::<f32>()?;
        let dims: Vec<i64> = shape.to_vec();
        if dims.len() != 3 || dims[2] as usize != f0::RMVPE_CLASSES {
            bail!("RMVPE の出力形状が想定外です: {:?}", dims);
        }
        let got_frames = dims[1] as usize;
        let use_frames = frames.min(got_frames);
        let mut f0 = f0::decode_rmvpe(
            &data[..use_frames * f0::RMVPE_CLASSES],
            use_frames,
            opts.f0_threshold,
        );
        f0::shift_semitones(&mut f0, opts.pitch_semitones);
        f0.resize(p_len, 0.0);
        let coarse = f0::coarse(&f0);
        Ok((coarse, f0))
    }

    /// 1 区間の変換: ContentVec → 生成器。戻り値は tgt_sr の波形。
    fn vc(
        &mut self,
        seg: &[f32],
        pitch: &[i64],
        pitchf: &[f32],
        rng: &mut rand::rngs::StdRng,
    ) -> Result<Vec<f32>> {
        // ContentVec 特徴量 [1,T,768]
        let in_name = self.embedder.inputs()[0].name().to_string();
        let out_name = self.embedder.outputs()[0].name().to_string();
        let tensor = TensorRef::from_array_view(([1usize, seg.len()], seg))?;
        let mask = vec![1i64; seg.len()];
        let outputs = if self.embedder_has_mask {
            self.embedder.run(ort::inputs![
                in_name.as_str() => tensor,
                "attention_mask" => TensorRef::from_array_view(([1usize, seg.len()], mask.as_slice()))?,
            ])?
        } else {
            self.embedder
                .run(ort::inputs![in_name.as_str() => tensor])?
        };
        let (shape, feats) = outputs[out_name.as_str()].try_extract_tensor::<f32>()?;
        let dims: Vec<i64> = shape.to_vec();
        if dims.len() != 3 {
            bail!("ContentVec の出力形状が想定外です: {:?}", dims);
        }
        let t_feat = dims[1] as usize;
        let ch = dims[2] as usize;
        if ch != 768 {
            bail!("ContentVec の特徴量次元が 768 ではありません: {ch}");
        }
        // 2 倍に引き伸ばし（nearest）
        let l = (t_feat * 2).min(pitch.len()).min(pitchf.len());
        if l == 0 {
            bail!("特徴量フレームが 0 です");
        }
        let mut phone = vec![0.0f32; l * ch];
        for i in 0..l {
            let src = &feats[(i / 2) * ch..(i / 2 + 1) * ch];
            phone[i * ch..(i + 1) * ch].copy_from_slice(src);
        }
        let pitch_v: Vec<i64> = pitch[..l].to_vec();
        let pitchf_v: Vec<f32> = pitchf[..l].to_vec();
        let lengths = [l as i64];
        let ds = [0i64];
        let rnd: Vec<f32> = (0..LATENT_CHANNELS * l)
            .map(|_| rng.sample::<f32, _>(StandardNormal))
            .collect();
        drop(outputs);

        let gen = self
            .generator
            .as_mut()
            .ok_or_else(|| anyhow!("声モデルが未読み込みです"))?;
        let outputs = if gen.needs_f0 {
            gen.session.run(ort::inputs![
                "phone" => TensorRef::from_array_view(([1usize, l, ch], phone.as_slice()))?,
                "phone_lengths" => TensorRef::from_array_view(([1usize], lengths.as_slice()))?,
                "pitch" => TensorRef::from_array_view(([1usize, l], pitch_v.as_slice()))?,
                "pitchf" => TensorRef::from_array_view(([1usize, l], pitchf_v.as_slice()))?,
                "ds" => TensorRef::from_array_view(([1usize], ds.as_slice()))?,
                "rnd" => TensorRef::from_array_view(([1usize, LATENT_CHANNELS, l], rnd.as_slice()))?,
            ])?
        } else {
            gen.session.run(ort::inputs![
                "phone" => TensorRef::from_array_view(([1usize, l, ch], phone.as_slice()))?,
                "phone_lengths" => TensorRef::from_array_view(([1usize], lengths.as_slice()))?,
                "ds" => TensorRef::from_array_view(([1usize], ds.as_slice()))?,
                "rnd" => TensorRef::from_array_view(([1usize, LATENT_CHANNELS, l], rnd.as_slice()))?,
            ])?
        };
        let (_, audio) = outputs["audio"].try_extract_tensor::<f32>()?;
        Ok(audio.to_vec())
    }
}

fn push_trimmed(out: &mut Vec<f32>, y: &[f32], trim: usize) {
    if y.len() > 2 * trim {
        out.extend_from_slice(&y[trim..y.len() - trim]);
    }
}

/// numpy.pad(mode="reflect") 相当。
fn reflect_pad(x: &[f32], left: usize, right: usize) -> Vec<f32> {
    let n = x.len();
    let mut v = Vec::with_capacity(n + left + right);
    for i in 0..left {
        let idx = reflect_index(-(left as isize) + i as isize, n);
        v.push(x[idx]);
    }
    v.extend_from_slice(x);
    for i in 0..right {
        let idx = reflect_index(n as isize + i as isize, n);
        v.push(x[idx]);
    }
    v
}

fn reflect_index(i: isize, n: usize) -> usize {
    if n == 1 {
        return 0;
    }
    let period = 2 * (n as isize - 1);
    let mut k = i.rem_euclid(period);
    if k >= n as isize {
        k = period - k;
    }
    k as usize
}

/// librosa.feature.rms（center=True, 定数ゼロ埋め）。
fn rms_frames(x: &[f32], frame_length: usize, hop: usize) -> Vec<f32> {
    let pad = frame_length / 2;
    let n = x.len();
    let frames = 1 + n / hop;
    let mut out = Vec::with_capacity(frames);
    for f in 0..frames {
        let start = f * hop;
        let mut acc = 0.0f64;
        for i in 0..frame_length {
            let idx = start + i;
            if idx >= pad && idx - pad < n {
                let v = x[idx - pad] as f64;
                acc += v * v;
            }
        }
        out.push((acc / frame_length as f64).sqrt() as f32);
    }
    out
}

fn interp_linear(v: &[f32], len: usize) -> Vec<f32> {
    if v.len() <= 1 {
        return vec![v.first().copied().unwrap_or(0.0); len];
    }
    (0..len)
        .map(|i| {
            let pos = i as f64 * (v.len() - 1) as f64 / (len.max(2) - 1) as f64;
            let k = pos.floor() as usize;
            let frac = (pos - k as f64) as f32;
            if k + 1 < v.len() {
                v[k] * (1.0 - frac) + v[k + 1] * frac
            } else {
                v[k]
            }
        })
        .collect()
}

/// 入力の音量エンベロープを出力に混ぜる（RVC の change_rms）。
fn change_rms(input: &[f32], sr_in: u32, output: &mut [f32], sr_out: u32, rate: f32) {
    let r1 = rms_frames(input, (sr_in as usize / 2) * 2, sr_in as usize / 2);
    let r2 = rms_frames(output, (sr_out as usize / 2) * 2, sr_out as usize / 2);
    let r1 = interp_linear(&r1, output.len());
    let r2 = interp_linear(&r2, output.len());
    for i in 0..output.len() {
        let a = r1[i].max(1e-6).powf(1.0 - rate);
        let b = r2[i].max(1e-6).powf(rate - 1.0);
        output[i] *= a * b;
    }
}

fn validate_embedder(s: &Session) -> Result<()> {
    if s.inputs().is_empty() || s.outputs().is_empty() {
        bail!("ContentVec モデルの入出力が空です");
    }
    Ok(())
}

fn validate_rmvpe(s: &Session) -> Result<()> {
    if s.inputs().is_empty() || s.outputs().is_empty() {
        bail!("RMVPE モデルの入出力が空です");
    }
    Ok(())
}

/// 生成器の入出力を検証し、F0（pitch/pitchf）入力を持つかを返す。
fn validate_generator(s: &Session) -> Result<bool> {
    let names: Vec<&str> = s.inputs().iter().map(|o| o.name()).collect();
    for need in ["phone", "phone_lengths", "ds", "rnd"] {
        if !names.contains(&need) {
            bail!("声モデルの入力 `{need}` がありません（入力: {:?}）", names);
        }
    }
    let has_pitch = names.contains(&"pitch");
    let has_pitchf = names.contains(&"pitchf");
    if has_pitch != has_pitchf {
        bail!(
            "声モデルの pitch/pitchf 入力が片方しかありません（入力: {:?}）",
            names
        );
    }
    if !s.outputs().iter().any(|o| o.name() == "audio") {
        bail!("声モデルの出力 `audio` がありません");
    }
    Ok(has_pitch)
}

/// vc-convert が書くメタデータ `metadata` = {"f0":true,"samplingRate":40000,"version":"v2"} から SR を読む。
fn generator_sample_rate(s: &Session) -> Option<u32> {
    let m = s.metadata().ok()?;
    let json = m.custom("metadata")?;
    let v: serde_json::Value = serde_json::from_str(&json).ok()?;
    v.get("samplingRate")?.as_u64().map(|x| x as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reflect_pad_matches_numpy() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let y = reflect_pad(&x, 2, 3);
        assert_eq!(y, vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn interp_endpoints() {
        let v = [0.0, 10.0];
        let y = interp_linear(&v, 5);
        assert_eq!(y, vec![0.0, 2.5, 5.0, 7.5, 10.0]);
    }
}
