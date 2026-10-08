//! 同梱する 5 つの声モデル（愛想良い系少女の声セット V2）。

use crate::paths::Home;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Voice {
    #[default]
    Sarasara,
    Howatto,
    Kakkoii,
    Sittori,
    Syakitto,
}

impl Voice {
    pub const ALL: [Voice; 5] = [
        Voice::Sarasara,
        Voice::Howatto,
        Voice::Kakkoii,
        Voice::Sittori,
        Voice::Syakitto,
    ];

    /// CLI で指定する短い識別子。
    pub fn id(self) -> &'static str {
        match self {
            Voice::Sarasara => "sarasara",
            Voice::Howatto => "howatto",
            Voice::Kakkoii => "kakkoii",
            Voice::Sittori => "sittori",
            Voice::Syakitto => "syakitto",
        }
    }

    /// GUI に表示する日本語名。
    pub fn display_ja(self) -> &'static str {
        match self {
            Voice::Sarasara => "さらさら風味ちゃん",
            Voice::Howatto => "ほわっと風味ちゃん",
            Voice::Kakkoii => "かっこいい風味ちゃん",
            Voice::Sittori => "しっとり風味ちゃん",
            Voice::Syakitto => "しゃきっと風味ちゃん",
        }
    }

    /// `models/` 内のファイル名（拡張子なし）。
    pub fn file_stem(self) -> &'static str {
        match self {
            Voice::Sarasara => "V2-AISO-SARASARA",
            Voice::Howatto => "V2-AISO-HOWATTO",
            Voice::Kakkoii => "V2-AISO-KAKKOII",
            Voice::Sittori => "V2-AISO-SITTORI",
            Voice::Syakitto => "V2-AISO-SYAKITTO",
        }
    }

    pub fn parse(s: &str) -> Option<Voice> {
        let s = s.trim().to_ascii_lowercase();
        Voice::ALL
            .iter()
            .copied()
            .find(|v| v.id() == s || v.file_stem().to_ascii_lowercase() == s)
    }

    pub fn pth_path(self, home: &Home) -> PathBuf {
        home.models_dir().join(format!("{}.pth", self.file_stem()))
    }

    pub fn onnx_path(self, home: &Home) -> PathBuf {
        home.models_dir().join(format!("{}.onnx", self.file_stem()))
    }
}

impl std::fmt::Display for Voice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.id())
    }
}
