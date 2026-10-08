//! boin-core: RVC（Retrieval-based Voice Conversion）v2 モデルを ONNX Runtime で動かす
//! 音声ファイル変換パイプライン。Python / PyTorch を実行時に必要としない。
//!
//! 処理の流れ（RVC WebUI の非リアルタイム推論経路を再現）:
//! 入力音声 → 16kHz モノラル → ハイパス → 無音境界で分割 →
//! ContentVec（特徴量）+ RMVPE（基本周波数）→ 生成器（ONNX 化した .pth）→ 40kHz 音声

pub mod audio;
pub mod dsp;
pub mod f0;
pub mod models;
pub mod onnx;
pub mod paths;
pub mod pipeline;
pub mod prepare;

pub use models::Voice;
pub use onnx::Device;
pub use paths::Home;
pub use pipeline::{ConvertOptions, ConvertReport, Converter, Stage};
