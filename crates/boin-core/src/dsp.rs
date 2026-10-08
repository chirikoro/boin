//! 信号処理: ハイパスフィルタ（filtfilt）、STFT、mel フィルタバンク（librosa 互換）、log-mel。

use realfft::num_complex::Complex;
use realfft::{RealFftPlanner, RealToComplex};
use std::sync::Arc;

/// RVC 公式と同じ 5 次バターワース・ハイパス（48Hz, fs=16kHz）。scipy.signal.butter で算出した係数。
pub const HP_B: [f64; 6] = [
    0.9699606451838447,
    -4.849803225919223,
    9.699606451838447,
    -9.699606451838447,
    4.849803225919223,
    -0.9699606451838447,
];
pub const HP_A: [f64; 6] = [
    1.0,
    -4.939001819168364,
    9.757863526739543,
    -9.639544849413458,
    4.761506797356209,
    -0.9408236532054606,
];
/// scipy.signal.lfilter_zi(HP_B, HP_A)
const HP_ZI: [f64; 5] = [
    -0.9699607413707367,
    3.879842959615721,
    -5.81976443080129,
    3.879842948235015,
    -0.9699607356787477,
];

/// 直接型 II 転置構造の IIR フィルタ（初期状態 zi を x[0] 倍して適用）。
fn lfilter(b: &[f64; 6], a: &[f64; 6], x: &[f64], x0: f64) -> Vec<f64> {
    let mut z = [0.0f64; 5];
    for i in 0..5 {
        z[i] = HP_ZI[i] * x0;
    }
    let mut y = Vec::with_capacity(x.len());
    for &xn in x {
        let yn = b[0] * xn + z[0];
        for i in 0..4 {
            z[i] = b[i + 1] * xn + z[i + 1] - a[i + 1] * yn;
        }
        z[4] = b[5] * xn - a[5] * yn;
        y.push(yn);
    }
    y
}

/// scipy.signal.filtfilt（padtype="odd", padlen=3*max(len(a),len(b))=18）相当のゼロ位相ハイパス。
pub fn highpass_filtfilt(x: &[f32]) -> Vec<f32> {
    let padlen = 18usize;
    if x.len() <= padlen {
        return x.to_vec();
    }
    let n = x.len();
    let xd: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    // odd 拡張
    let mut ext = Vec::with_capacity(n + 2 * padlen);
    for i in (1..=padlen).rev() {
        ext.push(2.0 * xd[0] - xd[i]);
    }
    ext.extend_from_slice(&xd);
    for i in 1..=padlen {
        ext.push(2.0 * xd[n - 1] - xd[n - 1 - i]);
    }
    let fwd = lfilter(&HP_B, &HP_A, &ext, ext[0]);
    let mut rev: Vec<f64> = fwd.iter().rev().copied().collect();
    let bwd = lfilter(&HP_B, &HP_A, &rev, rev[0]);
    rev.clear();
    rev.extend(bwd.iter().rev());
    rev[padlen..padlen + n].iter().map(|&v| v as f32).collect()
}

/// torch.stft(center=True, pad_mode="reflect", window=hann_window(periodic)) と同じ振幅スペクトログラム。
pub struct Stft {
    n_fft: usize,
    hop: usize,
    window: Vec<f32>,
    fft: Arc<dyn RealToComplex<f32>>,
}

impl Stft {
    pub fn new(n_fft: usize, hop: usize) -> Stft {
        let window: Vec<f32> = (0..n_fft)
            .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / n_fft as f32).cos())
            .collect();
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(n_fft);
        Stft {
            n_fft,
            hop,
            window,
            fft,
        }
    }

    pub fn n_bins(&self) -> usize {
        self.n_fft / 2 + 1
    }

    /// 反射パディング後のフレーム数 = 1 + len/hop。
    pub fn n_frames(&self, len: usize) -> usize {
        1 + len / self.hop
    }

    /// 振幅 |X| を [frames][bins] で返す（行優先 Vec）。
    pub fn magnitude(&self, x: &[f32]) -> (Vec<f32>, usize) {
        let pad = self.n_fft / 2;
        let n = x.len();
        // reflect パディング（torch と同じ: 端を含まない反射）
        let mut padded = Vec::with_capacity(n + 2 * pad);
        for i in (1..=pad).rev() {
            padded.push(x[i.min(n - 1)]);
        }
        padded.extend_from_slice(x);
        for i in 1..=pad {
            padded.push(x[n - 1 - i.min(n - 1)]);
        }
        let frames = self.n_frames(n);
        let bins = self.n_bins();
        let mut out = vec![0.0f32; frames * bins];
        let mut input = self.fft.make_input_vec();
        let mut spectrum = self.fft.make_output_vec();
        for f in 0..frames {
            let start = f * self.hop;
            for (i, (slot, w)) in input.iter_mut().zip(self.window.iter()).enumerate() {
                *slot = padded.get(start + i).copied().unwrap_or(0.0) * w;
            }
            self.fft.process(&mut input, &mut spectrum).expect("fft");
            let row = &mut out[f * bins..(f + 1) * bins];
            for (o, c) in row.iter_mut().zip(spectrum.iter()) {
                *o = Complex::norm(*c);
            }
        }
        (out, frames)
    }
}

fn hz_to_mel_slaney(f: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f64).ln() / 27.0;
    if f >= min_log_hz {
        min_log_mel + (f / min_log_hz).ln() / logstep
    } else {
        f / f_sp
    }
}

fn mel_to_hz_slaney(m: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f64).ln() / 27.0;
    if m >= min_log_mel {
        min_log_hz * (logstep * (m - min_log_mel)).exp()
    } else {
        f_sp * m
    }
}

/// librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax, htk=False, norm="slaney") 相当。[n_mels][bins]
pub fn mel_filterbank(sr: u32, n_fft: usize, n_mels: usize, fmin: f64, fmax: f64) -> Vec<f32> {
    let bins = n_fft / 2 + 1;
    let fftfreqs: Vec<f64> = (0..bins)
        .map(|i| i as f64 * sr as f64 / n_fft as f64)
        .collect();
    let m_min = hz_to_mel_slaney(fmin);
    let m_max = hz_to_mel_slaney(fmax);
    let mel_f: Vec<f64> = (0..n_mels + 2)
        .map(|i| mel_to_hz_slaney(m_min + (m_max - m_min) * i as f64 / (n_mels + 1) as f64))
        .collect();
    let mut weights = vec![0.0f32; n_mels * bins];
    for i in 0..n_mels {
        let fdiff_lo = mel_f[i + 1] - mel_f[i];
        let fdiff_hi = mel_f[i + 2] - mel_f[i + 1];
        let enorm = 2.0 / (mel_f[i + 2] - mel_f[i]);
        for j in 0..bins {
            let lower = (fftfreqs[j] - mel_f[i]) / fdiff_lo;
            let upper = (mel_f[i + 2] - fftfreqs[j]) / fdiff_hi;
            let w = lower.min(upper).max(0.0);
            weights[i * bins + j] = (w * enorm) as f32;
        }
    }
    weights
}

/// RMVPE 用 log-mel 抽出器（n_fft 1024, hop 160, 128 mel, 30–8000Hz, clamp 1e-5）。
pub struct LogMel {
    stft: Stft,
    fb: Vec<f32>,
    n_mels: usize,
}

impl LogMel {
    pub fn rmvpe() -> LogMel {
        let stft = Stft::new(1024, 160);
        let fb = mel_filterbank(16000, 1024, 128, 30.0, 8000.0);
        LogMel {
            stft,
            fb,
            n_mels: 128,
        }
    }

    pub fn n_mels(&self) -> usize {
        self.n_mels
    }

    /// 返り値は [n_mels][frames]（行優先）とフレーム数。
    pub fn compute(&self, x: &[f32]) -> (Vec<f32>, usize) {
        let (mag, frames) = self.stft.magnitude(x);
        let bins = self.stft.n_bins();
        let mut out = vec![0.0f32; self.n_mels * frames];
        for m in 0..self.n_mels {
            let row = &self.fb[m * bins..(m + 1) * bins];
            // 非ゼロ区間だけ使う
            let first = row.iter().position(|&w| w > 0.0).unwrap_or(0);
            let last = row
                .iter()
                .rposition(|&w| w > 0.0)
                .map(|i| i + 1)
                .unwrap_or(0);
            for f in 0..frames {
                let spec = &mag[f * bins..(f + 1) * bins];
                let mut acc = 0.0f32;
                for j in first..last {
                    acc += row[j] * spec[j];
                }
                out[m * frames + f] = acc.max(1e-5).ln();
            }
        }
        (out, frames)
    }
}

#[cfg(test)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;

    #[test]
    fn mel_matches_librosa_reference() {
        // librosa.filters.mel(sr=16000, n_fft=1024, n_mels=128, fmin=30, fmax=8000) の参照値
        let fb = mel_filterbank(16000, 1024, 128, 30.0, 8000.0);
        let bins = 513;
        let row0: f32 = fb[0..bins].iter().sum();
        let row1: f32 = fb[bins..2 * bins].iter().sum();
        assert!((row0 - 0.059569891542196274).abs() < 1e-5, "{row0}");
        assert!((row1 - 0.06787431240081787).abs() < 1e-5, "{row1}");
        let (idx0, max0) = fb[0..bins]
            .iter()
            .enumerate()
            .fold((0, 0.0f32), |a, (i, &v)| if v > a.1 { (i, v) } else { a });
        assert_eq!(idx0, 3);
        assert!((max0 - 0.03148721158504486).abs() < 1e-5);
        let r127 = &fb[127 * bins..128 * bins];
        let (idx, mx) = r127
            .iter()
            .enumerate()
            .fold((0, 0.0f32), |a, (i, &v)| if v > a.1 { (i, v) } else { a });
        assert_eq!(idx, 500);
        assert!((mx - 0.005326752085238695).abs() < 1e-5);
        assert_eq!(r127.iter().filter(|&&v| v > 0.0).count(), 23);
    }

    #[test]
    fn highpass_removes_dc_keeps_speech_band() {
        let n = 16000;
        let x: Vec<f32> = (0..n)
            .map(|i| 0.5 + (2.0 * std::f32::consts::PI * 200.0 * i as f32 / 16000.0).sin())
            .collect();
        let y = highpass_filtfilt(&x);
        let mean: f32 = y[2000..14000].iter().sum::<f32>() / 12000.0;
        assert!(mean.abs() < 0.01, "mean={mean}");
        let amp = y[2000..14000].iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        assert!((amp - 1.0).abs() < 0.05, "amp={amp}");
    }

    #[test]
    fn stft_frame_count() {
        let s = Stft::new(1024, 160);
        let x = vec![0.0f32; 16000];
        let (_, frames) = s.magnitude(&x);
        assert_eq!(frames, 101);
    }
}
