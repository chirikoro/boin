//! 変換・準備処理をバックグラウンドスレッドで実行し、進捗を UI に送る。

use crate::settings::Settings;
use boin_core::prepare::{self, Progress};
use boin_core::{ConvertOptions, Converter, Home, Stage};
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

#[derive(Clone, Debug)]
pub enum FileStatus {
    Pending,
    Running(String),
    Done(String),
    Skipped(String),
    Failed(String),
}

pub enum Msg {
    Log(String),
    Progress { text: String, frac: Option<f32> },
    File { index: usize, status: FileStatus },
    Finished { summary: String },
}

pub struct Job {
    pub index: usize,
    pub input: PathBuf,
    pub out_dir: PathBuf,
}

fn prep_progress(tx: &Sender<Msg>, p: Progress) {
    match p {
        Progress::Download { file, done, total } => {
            let frac = if total > 0 {
                Some(done as f32 / total as f32)
            } else {
                None
            };
            let _ = tx.send(Msg::Progress {
                text: format!(
                    "ダウンロード中 {file}: {:.0} / {:.0} MB",
                    done as f64 / 1e6,
                    total as f64 / 1e6
                ),
                frac,
            });
        }
        Progress::Verify { file } => {
            let _ = tx.send(Msg::Progress {
                text: format!("{file} を検証中"),
                frac: None,
            });
        }
        Progress::Convert { voice, stage } => {
            let _ = tx.send(Msg::Progress {
                text: format!("{} を変換中: {stage}", voice.display_ja()),
                frac: None,
            });
        }
        Progress::Done { what } => {
            let _ = tx.send(Msg::Log(format!("完了: {what}")));
        }
    }
}

pub fn spawn_prepare(home: Home, settings: Settings) -> Receiver<Msg> {
    let (tx, rx) = channel();
    std::thread::spawn(move || {
        let mut cb = |p: Progress| prep_progress(&tx, p);
        let result = prepare::ensure_base_models(&home, settings.lite, false, &mut cb)
            .and_then(|_| prepare::ensure_voice_onnx(&home, settings.voice, &mut cb).map(|_| ()));
        let summary = match result {
            Ok(()) => "モデルの準備が完了しました".to_owned(),
            Err(e) => {
                let _ = tx.send(Msg::Log(format!("エラー: {e:#}")));
                format!("準備に失敗しました: {e}")
            }
        };
        let _ = tx.send(Msg::Finished { summary });
    });
    rx
}

pub fn spawn_convert(
    home: Home,
    settings: Settings,
    jobs: Vec<Job>,
    cancel: Arc<AtomicBool>,
) -> Receiver<Msg> {
    let (tx, rx) = channel();
    std::thread::spawn(move || {
        let summary = run_convert(&home, &settings, &jobs, &cancel, &tx);
        let _ = tx.send(Msg::Finished { summary });
    });
    rx
}

fn run_convert(
    home: &Home,
    settings: &Settings,
    jobs: &[Job],
    cancel: &AtomicBool,
    tx: &Sender<Msg>,
) -> String {
    let mut cb = |p: Progress| prep_progress(tx, p);
    if let Err(e) = prepare::ensure_base_models(home, settings.lite, false, &mut cb) {
        let _ = tx.send(Msg::Log(format!("エラー: {e:#}")));
        return format!("基盤モデルの準備に失敗しました: {e}");
    }
    let threads = boin_core::onnx::default_threads();
    let mut conv = match Converter::new(home, settings.device, settings.lite, threads) {
        Ok(c) => c,
        Err(e) => {
            let _ = tx.send(Msg::Log(format!("エラー: {e:#}")));
            return format!("モデルの読み込みに失敗しました: {e}");
        }
    };
    let _ = tx.send(Msg::Progress {
        text: format!("{} を読み込み中…", settings.voice.display_ja()),
        frac: None,
    });
    if let Err(e) = conv.load_voice(settings.voice, &mut cb) {
        let _ = tx.send(Msg::Log(format!("エラー: {e:#}")));
        return format!("声モデルの読み込みに失敗しました: {e}");
    }
    let opts = ConvertOptions {
        voice: settings.voice,
        output_sample_rate: settings.output_sr,
        ..Default::default()
    };
    let total = jobs.len();
    let mut ok = 0usize;
    let mut failed = 0usize;
    let mut skipped = 0usize;
    for (n, job) in jobs.iter().enumerate() {
        if cancel.load(Ordering::SeqCst) {
            let _ = tx.send(Msg::Log("中止しました".to_owned()));
            return format!("中止: {ok} 件完了, {failed} 件失敗, {} 件未処理", total - n);
        }
        let stem = job
            .input
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("output");
        let output = job
            .out_dir
            .join(format!("{stem}_{}.wav", settings.voice.id()));
        if output.exists() && !settings.overwrite {
            skipped += 1;
            let _ = tx.send(Msg::File {
                index: job.index,
                status: FileStatus::Skipped("既存のためスキップ".to_owned()),
            });
            continue;
        }
        let base = n as f32 / total as f32;
        let _ = tx.send(Msg::File {
            index: job.index,
            status: FileStatus::Running("変換中".to_owned()),
        });
        let name = job
            .input
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_owned();
        let mut stage_cb = |s: Stage| {
            let (text, sub) = match s {
                Stage::LoadModel(_) => ("モデル読込".to_owned(), 0.0),
                Stage::Decode => ("デコード".to_owned(), 0.05),
                Stage::Pitch => ("ピッチ推定".to_owned(), 0.15),
                Stage::Segment { index, total: t } => (
                    format!("変換 {index}/{t}"),
                    0.2 + 0.75 * (index as f32 - 1.0) / t as f32,
                ),
                Stage::Write => ("書き出し".to_owned(), 0.97),
            };
            let _ = tx.send(Msg::File {
                index: job.index,
                status: FileStatus::Running(text.clone()),
            });
            let _ = tx.send(Msg::Progress {
                text: format!("[{}/{}] {name}: {text}", n + 1, total),
                frac: Some(base + sub / total as f32),
            });
        };
        match conv.convert_file(&job.input, &output, &opts, &mut stage_cb) {
            Ok(r) => {
                ok += 1;
                let msg = format!(
                    "完了 {:.1}s → {:.1}s ({:.2}x)",
                    r.input_secs,
                    r.elapsed.as_secs_f64(),
                    r.realtime_factor()
                );
                let _ = tx.send(Msg::Log(format!(
                    "{} → {} {msg}",
                    job.input.display(),
                    output.display()
                )));
                let _ = tx.send(Msg::File {
                    index: job.index,
                    status: FileStatus::Done(msg),
                });
            }
            Err(e) => {
                failed += 1;
                let _ = tx.send(Msg::Log(format!("失敗 {}: {e:#}", job.input.display())));
                let _ = tx.send(Msg::File {
                    index: job.index,
                    status: FileStatus::Failed(format!("失敗: {e}")),
                });
            }
        }
    }
    format!("完了: {ok} 件成功, {failed} 件失敗, {skipped} 件スキップ")
}
