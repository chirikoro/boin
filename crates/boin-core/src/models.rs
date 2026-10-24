//! 声モデルの定義。
//!
//! 同梱の 5 モデル（愛想良い系少女の声セット V2）に加えて、`models/` フォルダに置かれた
//! 任意の RVC v2 `.pth`（自前で学習・合成したモデルなど）も選べる。

use crate::paths::Home;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::path::{Path, PathBuf};

/// 声モデル 1 つ分。`id` は CLI 用の識別子、`stem` は `models/` 内のファイル名（拡張子なし）。
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Voice {
    id: String,
    display: String,
    stem: String,
}

/// 同梱モデルの (id, 表示名, ファイル名) 一覧。
const BUILTIN: [(&str, &str, &str); 5] = [
    ("sarasara", "さらさら風味ちゃん", "V2-AISO-SARASARA"),
    ("howatto", "ほわっと風味ちゃん", "V2-AISO-HOWATTO"),
    ("kakkoii", "かっこいい風味ちゃん", "V2-AISO-KAKKOII"),
    ("sittori", "しっとり風味ちゃん", "V2-AISO-SITTORI"),
    ("syakitto", "しゃきっと風味ちゃん", "V2-AISO-SYAKITTO"),
];

impl Voice {
    fn new(id: &str, display: &str, stem: &str) -> Voice {
        Voice {
            id: id.to_owned(),
            display: display.to_owned(),
            stem: stem.to_owned(),
        }
    }

    /// 同梱の 5 モデル。
    pub fn builtin() -> Vec<Voice> {
        BUILTIN
            .iter()
            .map(|(id, disp, stem)| Voice::new(id, disp, stem))
            .collect()
    }

    /// `models/` にあるファイル名から任意モデルを作る（表示名・id はファイル名そのまま）。
    pub fn custom(stem: &str) -> Voice {
        Voice::new(stem, stem, stem)
    }

    /// 同梱モデルか（`models/` に後から置いた任意モデルではないか）。
    pub fn is_builtin(&self) -> bool {
        BUILTIN.iter().any(|(_, _, stem)| *stem == self.stem)
    }

    /// 同梱 5 モデル＋`models/` 内の追加 `.pth` を列挙する（追加分はファイル名順）。
    pub fn discover(home: &Home) -> Vec<Voice> {
        let mut voices = Voice::builtin();
        let mut extra: Vec<String> = std::fs::read_dir(home.models_dir())
            .map(|rd| {
                rd.filter_map(|e| e.ok())
                    .map(|e| e.path())
                    .filter(|p| p.is_file() && has_ext(p, "pth"))
                    .filter_map(|p| p.file_stem()?.to_str().map(str::to_owned))
                    .filter(|stem| !BUILTIN.iter().any(|(_, _, s)| s == stem))
                    .collect()
            })
            .unwrap_or_default();
        extra.sort();
        extra.dedup();
        voices.extend(extra.iter().map(|s| Voice::custom(s)));
        voices
    }

    /// CLI で指定する短い識別子（同梱モデルは英小文字名、追加モデルはファイル名）。
    pub fn id(&self) -> &str {
        &self.id
    }

    /// GUI に表示する名前。
    pub fn display_ja(&self) -> &str {
        &self.display
    }

    /// `models/` 内のファイル名（拡張子なし）。
    pub fn file_stem(&self) -> &str {
        &self.stem
    }

    /// id またはファイル名から解決する（大文字小文字は区別しない）。
    /// 一覧に無くても `models/<名前>.pth` が存在すれば追加モデルとして受け付ける。
    pub fn parse(home: &Home, s: &str) -> Option<Voice> {
        let want = s.trim().trim_end_matches(".pth");
        let lower = want.to_ascii_lowercase();
        if let Some(v) = Voice::discover(home)
            .into_iter()
            .find(|v| v.id.to_ascii_lowercase() == lower || v.stem.to_ascii_lowercase() == lower)
        {
            return Some(v);
        }
        let p = home.models_dir().join(format!("{want}.pth"));
        p.is_file().then(|| Voice::custom(want))
    }

    /// id だけから復元する（設定ファイル用）。同梱モデルの id なら同梱、そうでなければ追加モデル扱い。
    pub fn from_id(id: &str) -> Voice {
        let lower = id.trim().to_ascii_lowercase();
        BUILTIN
            .iter()
            .find(|(i, _, stem)| *i == lower || stem.to_ascii_lowercase() == lower)
            .map(|(i, d, s)| Voice::new(i, d, s))
            .unwrap_or_else(|| Voice::custom(id.trim()))
    }

    pub fn pth_path(&self, home: &Home) -> PathBuf {
        home.models_dir().join(format!("{}.pth", self.stem))
    }

    pub fn onnx_path(&self, home: &Home) -> PathBuf {
        home.models_dir().join(format!("{}.onnx", self.stem))
    }
}

fn has_ext(p: &Path, ext: &str) -> bool {
    p.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case(ext))
}

impl Default for Voice {
    fn default() -> Self {
        let (id, disp, stem) = BUILTIN[0];
        Voice::new(id, disp, stem)
    }
}

impl std::fmt::Display for Voice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.id)
    }
}

/// 設定ファイルには id（文字列）だけを保存する。
impl Serialize for Voice {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.id)
    }
}

impl<'de> Deserialize<'de> for Voice {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let id = String::deserialize(d)?;
        Ok(Voice::from_id(&id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_roundtrip() {
        for v in Voice::builtin() {
            assert_eq!(Voice::from_id(v.id()), v);
            assert_eq!(Voice::from_id(v.file_stem()), v);
            assert!(v.is_builtin());
        }
    }

    #[test]
    fn custom_from_id() {
        let v = Voice::from_id("fem_blend");
        assert!(!v.is_builtin());
        assert_eq!(v.id(), "fem_blend");
        assert_eq!(v.file_stem(), "fem_blend");
    }
}
