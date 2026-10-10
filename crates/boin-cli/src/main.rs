//! boin CLI: 音声ファイルを RVC で変換する。

use anyhow::{bail, Context, Result};
use boin_core::prepare::{self, Progress};
use boin_core::{ConvertOptions, Converter, Device, Home, Stage, Voice};
use clap::{Parser, Subcommand};
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(
    name = "boin",
    version,
    about = "音声ファイルを架空の女性声（愛想良い系少女 V2）に変換します"
)]
struct Cli {
    /// models/ と assets/ があるフォルダ（既定: 実行ファイルの場所から自動検出）
    #[arg(long, global = true)]
    home: Option<PathBuf>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// 基盤モデルのダウンロードと声モデルの ONNX 変換を行う
    Setup {
        /// int8 量子化版の ContentVec を使う（軽量・やや低精度）
        #[arg(long)]
        lite: bool,
        /// 特定の声モデルだけ変換する（省略時は 5 つ全て）
        #[arg(short = 'm', long)]
        model: Option<String>,
        /// RMVPE（ピッチ推定）も取得する。同梱の 5 モデルはピッチ無しのため通常不要
        #[arg(long)]
        with_rmvpe: bool,
    },
    /// 音声ファイルを変換する
    Convert {
        /// 入力ファイル（複数可: wav/mp3/flac/ogg/m4a など）
        #[arg(required = true)]
        inputs: Vec<PathBuf>,
        /// 声モデル: sarasara / howatto / kakkoii / sittori / syakitto
        #[arg(short = 'm', long, default_value = "sarasara")]
        model: String,
        /// ピッチ変更（半音）。+12 で 1 オクターブ上
        #[arg(short = 'p', long, default_value_t = 0.0)]
        pitch: f32,
        /// 出力フォルダ（既定: 入力と同じ場所）
        #[arg(short = 'o', long)]
        out_dir: Option<PathBuf>,
        /// 推論デバイス: cpu / directml（Windows） / coreml（macOS）
        #[arg(long, default_value = "cpu")]
        device: String,
        /// 出力サンプルレート（既定: 40000）
        #[arg(long)]
        output_sr: Option<u32>,
        /// 音量エンベロープ混合率 0.0〜1.0（1.0 = 変換後そのまま）
        #[arg(long, default_value_t = 1.0)]
        rms_mix_rate: f32,
        /// int8 量子化版の ContentVec を使う
        #[arg(long)]
        lite: bool,
        /// スレッド数（既定: CPU コア数）
        #[arg(long)]
        threads: Option<usize>,
        /// 乱数シード（再現性が必要な場合）
        #[arg(long)]
        seed: Option<u64>,
        /// 既存の出力を上書きする
        #[arg(long)]
        overwrite: bool,
    },
    /// 声モデルと基盤モデルの準備状況を表示する
    Models,
    /// ONNX モデルの入出力とメタデータを表示する
    Inspect { path: PathBuf },
    /// 実行環境を診断する
    Doctor,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env().add_directive("warn".parse()?),
        )
        .with_target(false)
        .init();
    let cli = Cli::parse();
    let home = match &cli.home {
        Some(p) => Home::from_root(p),
        None => Home::detect(),
    };
    boin_core::onnx::init_runtime(&home)?;
    match cli.command {
        Command::Setup {
            lite,
            model,
            with_rmvpe,
        } => setup(&home, lite, with_rmvpe, model.as_deref()),
        Command::Convert {
            inputs,
            model,
            pitch,
            out_dir,
            device,
            output_sr,
            rms_mix_rate,
            lite,
            threads,
            seed,
            overwrite,
        } => {
            let voice = Voice::parse(&model).with_context(|| format!("不明な声モデル: {model}"))?;
            let device =
                Device::parse(&device).with_context(|| format!("不明なデバイス: {device}"))?;
            let opts = ConvertOptions {
                voice,
                pitch_semitones: pitch,
                output_sample_rate: output_sr,
                rms_mix_rate,
                seed,
                ..Default::default()
            };
            convert(
                &home,
                &inputs,
                out_dir.as_deref(),
                device,
                lite,
                threads,
                overwrite,
                &opts,
            )
        }
        Command::Models => models(&home),
        Command::Inspect { path } => inspect(&path),
        Command::Doctor => doctor(&home),
    }
}

fn print_progress(p: Progress) {
    match &p {
        Progress::Download { file, done, total } => {
            let pct = if *total > 0 {
                *done as f64 * 100.0 / *total as f64
            } else {
                0.0
            };
            print!(
                "\r  {file}: {:.1} / {:.1} MB ({pct:.0}%)   ",
                *done as f64 / 1e6,
                *total as f64 / 1e6
            );
            let _ = std::io::stdout().flush();
        }
        Progress::Verify { file } => println!("\n  {file}: SHA-256 を検証中"),
        Progress::Convert { voice, stage } => println!("  {}: {stage}", voice.file_stem()),
        Progress::Done { what } => println!("  完了: {what}"),
    }
}

fn setup(home: &Home, lite: bool, with_rmvpe: bool, model: Option<&str>) -> Result<()> {
    println!("ホーム: {}", home.root.display());
    println!("[1/2] 基盤モデル");
    prepare::ensure_base_models(home, lite, with_rmvpe, &mut print_progress)?;
    println!("[2/2] 声モデルの ONNX 変換");
    let voices: Vec<Voice> = match model {
        Some(m) => vec![Voice::parse(m).with_context(|| format!("不明な声モデル: {m}"))?],
        None => Voice::ALL.to_vec(),
    };
    for v in voices {
        if !v.pth_path(home).is_file() {
            println!("  スキップ（.pth なし）: {}", v.pth_path(home).display());
            continue;
        }
        if prepare::voice_onnx_ready(home, v) {
            println!("  準備済み: {}", v.file_stem());
            continue;
        }
        prepare::ensure_voice_onnx(home, v, &mut print_progress)?;
    }
    println!("セットアップ完了");
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn convert(
    home: &Home,
    inputs: &[PathBuf],
    out_dir: Option<&Path>,
    device: Device,
    lite: bool,
    threads: Option<usize>,
    overwrite: bool,
    opts: &ConvertOptions,
) -> Result<()> {
    if !prepare::is_ready(home, opts.voice, lite) {
        println!("モデルが未準備のため、先にセットアップを行います。");
        setup(home, lite, false, Some(opts.voice.id()))?;
    }
    let threads = threads.unwrap_or_else(boin_core::onnx::default_threads);
    println!(
        "デバイス: {} / スレッド: {threads} / 声: {}",
        device.display_ja(),
        opts.voice.display_ja()
    );
    let mut conv = Converter::new(home, device, lite, threads)?;
    conv.load_voice(opts.voice, &mut print_progress)?;
    if conv.current_voice_uses_f0() == Some(false) && opts.pitch_semitones != 0.0 {
        eprintln!(
            "注意: {} はピッチ無しモデルのため --pitch は無視されます",
            opts.voice.display_ja()
        );
    }
    let mut failures = 0usize;
    for input in inputs {
        if !input.is_file() {
            eprintln!("見つかりません: {}", input.display());
            failures += 1;
            continue;
        }
        let output = output_path(input, out_dir, opts.voice);
        if output.exists() && !overwrite {
            eprintln!(
                "スキップ（既に存在。--overwrite で上書き）: {}",
                output.display()
            );
            continue;
        }
        println!("変換: {} → {}", input.display(), output.display());
        let mut last_line_len = 0usize;
        let mut show = |s: Stage| {
            let msg = match s {
                Stage::LoadModel(v) => format!("モデル読込 {}", v.file_stem()),
                Stage::Decode => "デコード".to_string(),
                Stage::Pitch => "ピッチ推定".to_string(),
                Stage::Segment { index, total } => format!("変換 {index}/{total}"),
                Stage::Write => "書き出し".to_string(),
            };
            print!("\r  {msg:<width$}", width = last_line_len.max(msg.len()));
            last_line_len = msg.chars().count();
            let _ = std::io::stdout().flush();
        };
        match conv.convert_file(input, &output, opts, &mut show) {
            Ok(r) => println!(
                "\r  完了: {:.1} 秒の音声を {:.1} 秒で変換（実時間比 {:.2}x, {} 区間, {} Hz）",
                r.input_secs,
                r.elapsed.as_secs_f64(),
                r.realtime_factor(),
                r.segments,
                r.output_sample_rate
            ),
            Err(e) => {
                println!();
                eprintln!("  失敗: {e:#}");
                failures += 1;
            }
        }
    }
    if failures > 0 {
        bail!("{failures} 件の変換に失敗しました");
    }
    Ok(())
}

fn output_path(input: &Path, out_dir: Option<&Path>, voice: Voice) -> PathBuf {
    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");
    let name = format!("{stem}_{}.wav", voice.id());
    match out_dir {
        Some(d) => d.join(name),
        None => input.with_file_name(name),
    }
}

fn models(home: &Home) -> Result<()> {
    println!("ホーム: {}", home.root.display());
    let st = prepare::status(home);
    println!("基盤モデル (assets/):");
    for (m, ok) in &st.base {
        println!(
            "  [{}] {} ({:.0} MB)",
            if *ok { "✓" } else { " " },
            m.file,
            m.bytes as f64 / 1e6
        );
    }
    println!("声モデル (models/):");
    for (v, pth, onnx) in &st.voices {
        println!(
            "  {:<9} {:<12} .pth:{} .onnx:{}",
            v.id(),
            v.display_ja(),
            if *pth { "✓" } else { "✗" },
            if *onnx { "✓" } else { "未変換" }
        );
    }
    Ok(())
}

fn inspect(path: &Path) -> Result<()> {
    let info = boin_core::onnx::inspect(path)?;
    println!("モデル: {}", path.display());
    if !info.producer.is_empty() {
        println!("producer: {}", info.producer);
    }
    if !info.description.is_empty() {
        println!("description: {}", info.description);
    }
    println!("入力:");
    for i in &info.inputs {
        println!("  {} : {}", i.name, i.dtype);
    }
    println!("出力:");
    for o in &info.outputs {
        println!("  {} : {}", o.name, o.dtype);
    }
    if !info.custom.is_empty() {
        println!("メタデータ:");
        for (k, v) in &info.custom {
            let v: String = v.chars().take(200).collect();
            println!("  {k} = {v}");
        }
    }
    Ok(())
}

fn doctor(home: &Home) -> Result<()> {
    println!("boin {}", env!("CARGO_PKG_VERSION"));
    println!("OS: {} / {}", std::env::consts::OS, std::env::consts::ARCH);
    println!("ONNX Runtime: {}", boin_core::onnx::runtime_version());
    if cfg!(windows) {
        match boin_core::onnx::init_runtime(home) {
            Ok(Some(p)) => println!("onnxruntime.dll: {}", p.display()),
            Ok(None) => {}
            Err(e) => println!("onnxruntime.dll: {e}"),
        }
    }
    println!(
        "利用可能デバイス: {}",
        Device::available()
            .iter()
            .map(|d| d.id())
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!("スレッド既定値: {}", boin_core::onnx::default_threads());
    println!("ホーム: {}", home.root.display());
    models(home)?;
    for m in [prepare::CONTENTVEC, prepare::CONTENTVEC_Q8, prepare::RMVPE] {
        if m.exists(home) {
            match prepare::verify_existing(home, &m) {
                Ok(true) => println!("  {}: SHA-256 OK", m.file),
                Ok(false) => println!("  {}: SHA-256 不一致（再ダウンロードを推奨）", m.file),
                Err(e) => println!("  {}: 検証エラー {e}", m.file),
            }
        }
    }
    Ok(())
}
