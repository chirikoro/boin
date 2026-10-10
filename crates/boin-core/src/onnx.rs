//! ONNX Runtime セッションの生成とモデル検査。

use anyhow::{Context, Result};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// 推論デバイス。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Device {
    #[default]
    Cpu,
    /// Windows の DirectML（DirectX 12 対応 GPU）。
    DirectMl,
    /// macOS の CoreML。
    CoreMl,
}

impl Device {
    pub fn id(self) -> &'static str {
        match self {
            Device::Cpu => "cpu",
            Device::DirectMl => "directml",
            Device::CoreMl => "coreml",
        }
    }

    pub fn display_ja(self) -> &'static str {
        match self {
            Device::Cpu => "CPU",
            Device::DirectMl => "GPU (DirectML)",
            Device::CoreMl => "GPU (CoreML)",
        }
    }

    pub fn parse(s: &str) -> Option<Device> {
        match s.trim().to_ascii_lowercase().as_str() {
            "cpu" => Some(Device::Cpu),
            "directml" | "dml" | "gpu" if cfg!(windows) => Some(Device::DirectMl),
            "coreml" | "gpu" if cfg!(target_os = "macos") => Some(Device::CoreMl),
            "directml" | "dml" => Some(Device::DirectMl),
            "coreml" => Some(Device::CoreMl),
            _ => None,
        }
    }

    /// この OS で選択できるデバイス一覧。
    pub fn available() -> Vec<Device> {
        let mut v = vec![Device::Cpu];
        if cfg!(windows) {
            v.push(Device::DirectMl);
        }
        if cfg!(target_os = "macos") {
            v.push(Device::CoreMl);
        }
        v
    }

    pub fn is_supported_here(self) -> bool {
        Device::available().contains(&self)
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.id())
    }
}

/// 既定のスレッド数（物理コア数の目安）。
pub fn default_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .clamp(1, 16)
}

/// ort のビルダー系エラー（ビルダー本体を内包し Send/Sync でない）を anyhow に変換する。
fn be<T, E: std::fmt::Display>(r: std::result::Result<T, E>) -> Result<T> {
    r.map_err(|e| anyhow::anyhow!("ONNX Runtime: {e}"))
}

/// セッションを作る。GPU 系デバイスの登録に失敗した場合は CPU にフォールバックする。
pub fn build_session(path: &Path, device: Device, threads: usize) -> Result<Session> {
    let builder = be(Session::builder())?;
    let builder = be(builder.with_optimization_level(GraphOptimizationLevel::Level3))?;
    let builder = be(builder.with_intra_threads(threads.max(1)))?;
    let builder = match device {
        Device::Cpu => builder,
        #[cfg(target_os = "macos")]
        Device::CoreMl => be(builder.with_execution_providers([ort::ep::CoreML::default()
            .with_subgraphs(true)
            .with_compute_units(ort::ep::coreml::ComputeUnits::All)
            .build()]))?,
        #[cfg(windows)]
        Device::DirectMl => {
            be(builder.with_execution_providers([ort::ep::DirectML::default().build()]))?
        }
        #[allow(unreachable_patterns)]
        other => {
            tracing::warn!("{} はこの OS では使えないため CPU で実行します", other.id());
            builder
        }
    };
    let mut builder = builder;
    builder
        .commit_from_file(path)
        .with_context(|| format!("ONNX モデルを読み込めません: {}", path.display()))
}

/// 入出力の名前と型。
#[derive(Clone, Debug)]
pub struct IoInfo {
    pub name: String,
    pub dtype: String,
}

#[derive(Clone, Debug)]
pub struct ModelInfo {
    pub inputs: Vec<IoInfo>,
    pub outputs: Vec<IoInfo>,
    pub producer: String,
    pub description: String,
    pub custom: Vec<(String, String)>,
}

/// ONNX モデルの入出力とメタデータを読む（CPU セッションを作って取得）。
pub fn inspect(path: &Path) -> Result<ModelInfo> {
    let session = build_session(path, Device::Cpu, 1)?;
    let io = |o: &ort::value::Outlet| IoInfo {
        name: o.name().to_string(),
        dtype: format!("{}", o.dtype()),
    };
    let inputs = session.inputs().iter().map(io).collect();
    let outputs = session.outputs().iter().map(io).collect();
    let (producer, description, custom) = match session.metadata() {
        Ok(m) => {
            let producer = m.producer().unwrap_or_default();
            let description = m.description().unwrap_or_default();
            let mut custom = Vec::new();
            if let Ok(keys) = m.custom_keys() {
                for k in keys {
                    if let Some(v) = m.custom(&k) {
                        custom.push((k, v));
                    }
                }
            }
            (producer, description, custom)
        }
        Err(_) => (String::new(), String::new(), Vec::new()),
    };
    Ok(ModelInfo {
        inputs,
        outputs,
        producer,
        description,
        custom,
    })
}

/// ONNX Runtime のバージョン文字列。
pub fn runtime_version() -> String {
    ort::info().to_string()
}

/// Windows で動的ロードする `onnxruntime.dll` のファイル名。
pub const ORT_DLL_NAME: &str = "onnxruntime.dll";

/// Windows: 実行ファイルの隣（または BOIN_HOME）にある `onnxruntime.dll` を明示的に読み込む。
/// 他の OS では何もしない。ONNX Runtime を使う前に 1 度呼ぶ。
pub fn init_runtime(home: &crate::paths::Home) -> Result<Option<std::path::PathBuf>> {
    #[cfg(windows)]
    {
        if let Ok(p) = std::env::var("ORT_DYLIB_PATH") {
            if !p.is_empty() {
                return Ok(Some(std::path::PathBuf::from(p)));
            }
        }
        let mut candidates: Vec<std::path::PathBuf> = Vec::new();
        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                candidates.push(dir.join(ORT_DLL_NAME));
            }
        }
        candidates.push(home.root.join(ORT_DLL_NAME));
        for c in &candidates {
            if c.is_file() {
                let builder = ort::init_from(c).map_err(|e| {
                    anyhow::anyhow!("{} の読み込みに失敗しました: {e}", c.display())
                })?;
                builder.with_name("boin").commit();
                return Ok(Some(c.clone()));
            }
        }
        anyhow::bail!(
            "onnxruntime.dll が見つかりません。次のいずれかに置いてください:\n{}\n\
             （Microsoft 公式の onnxruntime-win-x64-1.28.0.zip 内 lib/onnxruntime.dll）",
            candidates
                .iter()
                .map(|c| format!("  {}", c.display()))
                .collect::<Vec<_>>()
                .join("\n")
        )
    }
    #[cfg(not(windows))]
    {
        let _ = home;
        Ok(None)
    }
}
