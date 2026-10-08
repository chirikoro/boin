//! 実行ファイルの隣にある `models/` と `assets/` を探す。

use std::path::{Path, PathBuf};

/// アプリのホームディレクトリ（`models/` と `assets/` を含む場所）。
#[derive(Clone, Debug)]
pub struct Home {
    pub root: PathBuf,
}

impl Home {
    /// 環境変数 `BOIN_HOME` → 実行ファイルの場所とその親 → カレントディレクトリの順で探す。
    pub fn detect() -> Home {
        if let Ok(p) = std::env::var("BOIN_HOME") {
            return Home {
                root: PathBuf::from(p),
            };
        }
        let mut candidates: Vec<PathBuf> = Vec::new();
        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                candidates.push(dir.to_path_buf());
                for anc in dir.ancestors().skip(1).take(3) {
                    candidates.push(anc.to_path_buf());
                }
            }
        }
        if let Ok(cwd) = std::env::current_dir() {
            candidates.push(cwd);
        }
        for c in &candidates {
            if c.join("models").is_dir() {
                return Home { root: c.clone() };
            }
        }
        Home {
            root: candidates
                .last()
                .cloned()
                .unwrap_or_else(|| PathBuf::from(".")),
        }
    }

    pub fn from_root(root: impl AsRef<Path>) -> Home {
        Home {
            root: root.as_ref().to_path_buf(),
        }
    }

    pub fn models_dir(&self) -> PathBuf {
        self.root.join("models")
    }

    pub fn assets_dir(&self) -> PathBuf {
        self.root.join("assets")
    }
}
