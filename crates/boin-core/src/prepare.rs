//! モデルの準備: 基盤モデル（ContentVec / RMVPE）のダウンロードと、声モデル `.pth` → `.onnx` 変換。

use crate::models::Voice;
use crate::paths::Home;
use anyhow::{anyhow, bail, Context, Result};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

/// ダウンロードが必要な基盤モデル。
#[derive(Clone, Copy, Debug)]
pub struct BaseModel {
    pub key: &'static str,
    pub file: &'static str,
    pub url: &'static str,
    pub sha256: &'static str,
    pub bytes: u64,
}

/// ContentVec（HuBERT 系特徴量抽出器）。16kHz 波形 [1,N] → [1,T,768]。MIT ライセンス。
pub const CONTENTVEC: BaseModel = BaseModel {
    key: "contentvec",
    file: "contentvec_768l12.onnx",
    url: "https://huggingface.co/TigreGotico/voiceclonnx-rvc/resolve/main/contentvec_768l12.onnx",
    sha256: "cbddd8fa9352b3128df6359e2e3da6be0f9072e768e51e3efc3004e5030ea97f",
    bytes: 377_755_911,
};

/// ContentVec の int8 量子化版（軽量・やや低精度）。
pub const CONTENTVEC_Q8: BaseModel = BaseModel {
    key: "contentvec-q8",
    file: "contentvec_768l12_q8.onnx",
    url:
        "https://huggingface.co/TigreGotico/voiceclonnx-rvc/resolve/main/contentvec_768l12_q8.onnx",
    sha256: "23d4914b29779cd68af19191405423e82ec978fafe9c12227df23d3ebe04d421",
    bytes: 95224416,
};

/// RMVPE（基本周波数推定）。log-mel [1,128,T] → 顕著度 [1,T,360]。MIT ライセンス（RVC 公式配布）。
pub const RMVPE: BaseModel = BaseModel {
    key: "rmvpe",
    file: "rmvpe.onnx",
    url: "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.onnx",
    sha256: "5370e71ac80af8b4b7c793d27efd51fd8bf962de3a7ede0766dac0befa3660fd",
    bytes: 361_688_443,
};

impl BaseModel {
    pub fn path(&self, home: &Home) -> PathBuf {
        home.assets_dir().join(self.file)
    }
    pub fn exists(&self, home: &Home) -> bool {
        self.path(home).is_file()
    }
}

/// 使用する ContentVec モデル（lite = int8 版）。
pub fn embedder_model(lite: bool) -> BaseModel {
    if lite {
        CONTENTVEC_Q8
    } else {
        CONTENTVEC
    }
}

/// 準備処理の進捗通知。
#[derive(Clone, Debug)]
pub enum Progress {
    /// ダウンロード中（done / total バイト）。
    Download { file: String, done: u64, total: u64 },
    /// SHA-256 検証中。
    Verify { file: String },
    /// .pth → .onnx 変換中。
    Convert { voice: Voice, stage: String },
    /// 1 項目の完了。
    Done { what: String },
}

fn mtime(p: &Path) -> Option<u64> {
    fs::metadata(p)
        .ok()?
        .modified()
        .ok()?
        .duration_since(UNIX_EPOCH)
        .ok()
        .map(|d| d.as_secs())
}

/// 声モデルの ONNX が最新か（存在し、.pth より新しいか）。
pub fn voice_onnx_ready(home: &Home, voice: Voice) -> bool {
    let pth = voice.pth_path(home);
    let onnx = voice.onnx_path(home);
    match (onnx.is_file(), mtime(&pth), mtime(&onnx)) {
        (true, Some(p), Some(o)) => o >= p,
        (true, None, _) => true,
        _ => false,
    }
}

/// `.pth` から `.onnx` を生成（必要な場合のみ）。純 Rust（vc-convert）で変換する。
pub fn ensure_voice_onnx(
    home: &Home,
    voice: Voice,
    progress: &mut dyn FnMut(Progress),
) -> Result<PathBuf> {
    let onnx = voice.onnx_path(home);
    if voice_onnx_ready(home, voice) {
        return Ok(onnx);
    }
    let pth = voice.pth_path(home);
    if !pth.is_file() {
        bail!(
            "声モデルが見つかりません: {}\n（models/ フォルダに {}.pth を置いてください）",
            pth.display(),
            voice.file_stem()
        );
    }
    let options = vc_convert::ConvertOptions {
        export_mode: vc_convert::ExportMode::Webui,
        opset_version: 20,
    };
    let mut report = |stage: vc_convert::ProgressStage| {
        progress(Progress::Convert {
            voice,
            stage: stage.label().to_string(),
        })
    };
    let written = vc_convert::convert_pth_file(&pth, &options, &mut report)
        .with_context(|| format!("{} の ONNX 変換に失敗しました", pth.display()))?;
    if written != onnx {
        // vc-convert は入力の隣に <stem>.onnx を書く。想定と違えば移動する。
        fs::rename(&written, &onnx)
            .with_context(|| format!("{} → {} の移動に失敗", written.display(), onnx.display()))?;
    }
    progress(Progress::Done {
        what: format!("{} → ONNX", voice.file_stem()),
    });
    Ok(onnx)
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut f = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex::encode(hasher.finalize()))
}

/// URL からダウンロードし SHA-256 を検証して `dest` に置く（一時ファイル経由）。
pub fn download_verified(
    model: &BaseModel,
    dest: &Path,
    progress: &mut dyn FnMut(Progress),
) -> Result<()> {
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp = dest.with_extension("onnx.part");
    {
        let mut resp = ureq::get(model.url)
            .call()
            .with_context(|| format!("{} のダウンロード開始に失敗", model.url))?;
        let total = resp
            .headers()
            .get("content-length")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(model.bytes);
        let mut reader = resp.body_mut().with_config().limit(u64::MAX).reader();
        let mut out = fs::File::create(&tmp)?;
        let mut hasher = Sha256::new();
        let mut buf = vec![0u8; 1 << 20];
        let mut done = 0u64;
        let mut last_report = 0u64;
        loop {
            let n = reader.read(&mut buf)?;
            if n == 0 {
                break;
            }
            out.write_all(&buf[..n])?;
            hasher.update(&buf[..n]);
            done += n as u64;
            if done - last_report >= (1 << 20) {
                last_report = done;
                progress(Progress::Download {
                    file: model.file.to_string(),
                    done,
                    total,
                });
            }
        }
        out.flush()?;
        progress(Progress::Download {
            file: model.file.to_string(),
            done,
            total,
        });
        let got = hex::encode(hasher.finalize());
        progress(Progress::Verify {
            file: model.file.to_string(),
        });
        if got != model.sha256 {
            let _ = fs::remove_file(&tmp);
            bail!(
                "{} の SHA-256 が一致しません（期待 {}, 実際 {}）",
                model.file,
                model.sha256,
                got
            );
        }
    }
    fs::rename(&tmp, dest).with_context(|| format!("{} への移動に失敗", dest.display()))?;
    progress(Progress::Done {
        what: model.file.to_string(),
    });
    Ok(())
}

/// 基盤モデルがなければダウンロードする。`with_rmvpe` は F0 ありモデルを使う場合のみ必要。
pub fn ensure_base_models(
    home: &Home,
    lite: bool,
    with_rmvpe: bool,
    progress: &mut dyn FnMut(Progress),
) -> Result<()> {
    let mut list = vec![embedder_model(lite)];
    if with_rmvpe {
        list.push(RMVPE);
    }
    for m in list {
        let dest = m.path(home);
        if dest.is_file() {
            continue;
        }
        download_verified(&m, &dest, progress)?;
    }
    Ok(())
}

/// 既存のファイルの SHA-256 を検証する（`doctor` 用）。
pub fn verify_existing(home: &Home, model: &BaseModel) -> Result<bool> {
    let p = model.path(home);
    if !p.is_file() {
        return Err(anyhow!("{} がありません", p.display()));
    }
    Ok(sha256_file(&p)? == model.sha256)
}

/// 準備状況の一覧。
#[derive(Clone, Debug)]
pub struct Status {
    pub base: Vec<(BaseModel, bool)>,
    pub voices: Vec<(Voice, bool, bool)>, // (voice, pth あり, onnx 準備済み)
}

pub fn status(home: &Home) -> Status {
    Status {
        base: [CONTENTVEC, CONTENTVEC_Q8, RMVPE]
            .iter()
            .map(|m| (*m, m.exists(home)))
            .collect(),
        voices: Voice::ALL
            .iter()
            .map(|v| (*v, v.pth_path(home).is_file(), voice_onnx_ready(home, *v)))
            .collect(),
    }
}

/// 変換に必要なものが揃っているか（RMVPE は F0 ありモデルの読込時に遅延取得するため含めない）。
pub fn is_ready(home: &Home, voice: Voice, lite: bool) -> bool {
    embedder_model(lite).exists(home) && voice_onnx_ready(home, voice)
}
