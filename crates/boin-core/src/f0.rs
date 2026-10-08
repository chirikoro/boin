//! 基本周波数（F0）の後処理: RMVPE 出力のデコード、ピッチシフト、粗いピッチ量子化。

/// RMVPE の有声/無声判定しきい値（RVC 公式既定値）。
pub const RMVPE_THRESHOLD: f32 = 0.03;
pub const RMVPE_CLASSES: usize = 360;

/// cents_mapping = 20 * i + 1997.3794084376191（360 クラス）。
#[allow(clippy::excessive_precision)]
fn cents_of(i: usize) -> f32 {
    const OFFSET: f64 = 1997.3794084376191;
    (20.0 * i as f64 + OFFSET) as f32
}

/// RMVPE の顕著度 [frames][360] から F0(Hz) を求める（to_local_average_cents + decode）。
pub fn decode_rmvpe(salience: &[f32], frames: usize, thred: f32) -> Vec<f32> {
    let n = RMVPE_CLASSES;
    let mut f0 = Vec::with_capacity(frames);
    for f in 0..frames {
        let row = &salience[f * n..(f + 1) * n];
        let (center, maxv) =
            row.iter().enumerate().fold(
                (0usize, f32::MIN),
                |a, (i, &v)| if v > a.1 { (i, v) } else { a },
            );
        if maxv <= thred {
            f0.push(0.0);
            continue;
        }
        // 中心 ±4 の局所重み付き平均（範囲外はゼロ埋め）
        let mut product = 0.0f32;
        let mut weight = 0.0f32;
        let lo = center as isize - 4;
        for k in lo..=(center as isize + 4) {
            if k < 0 || k >= n as isize {
                continue;
            }
            let s = row[k as usize];
            product += s * cents_of(k as usize);
            weight += s;
        }
        let cents = if weight > 0.0 { product / weight } else { 0.0 };
        let hz = 10.0 * (2.0f32).powf(cents / 1200.0);
        f0.push(if (hz - 10.0).abs() < 1e-6 { 0.0 } else { hz });
    }
    f0
}

/// 半音単位のピッチシフト。
pub fn shift_semitones(f0: &mut [f32], semitones: f32) {
    if semitones == 0.0 {
        return;
    }
    let k = (2.0f32).powf(semitones / 12.0);
    for v in f0.iter_mut() {
        *v *= k;
    }
}

const F0_MIN: f32 = 50.0;
const F0_MAX: f32 = 1100.0;

/// RVC の粗いピッチ（1..=255 の mel 尺度量子化）。無声（0Hz）は 1。
pub fn coarse(f0: &[f32]) -> Vec<i64> {
    let mel_min = 1127.0 * (1.0 + F0_MIN / 700.0).ln();
    let mel_max = 1127.0 * (1.0 + F0_MAX / 700.0).ln();
    f0.iter()
        .map(|&hz| {
            let mut m = 1127.0 * (1.0 + hz / 700.0).ln();
            if m > 0.0 {
                m = (m - mel_min) * 254.0 / (mel_max - mel_min) + 1.0;
            }
            m = m.clamp(1.0, 255.0);
            m.round() as i64
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_peak_gives_expected_hz() {
        // クラス i に鋭いピークを置くと f0 = 10 * 2^(cents_of(i)/1200)
        let frames = 2;
        let mut s = vec![0.0f32; frames * RMVPE_CLASSES];
        s[100] = 1.0; // frame 0, class 100
                      // frame 1 は無声
        let f0 = decode_rmvpe(&s, frames, RMVPE_THRESHOLD);
        let expect = 10.0 * (2.0f32).powf(cents_of(100) / 1200.0);
        assert!((f0[0] - expect).abs() < 1e-2, "{} vs {}", f0[0], expect);
        assert_eq!(f0[1], 0.0);
    }

    #[test]
    fn coarse_range() {
        let c = coarse(&[0.0, 50.0, 220.0, 1100.0, 5000.0]);
        assert_eq!(c[0], 1);
        assert_eq!(c[1], 1);
        assert!(c[2] > 1 && c[2] < 255);
        assert_eq!(c[3], 255);
        assert_eq!(c[4], 255);
    }
}
