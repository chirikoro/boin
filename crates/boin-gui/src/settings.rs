//! 設定の永続化（config.toml）。

use boin_core::{Device, Voice};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub voice: Voice,
    pub out_dir: Option<PathBuf>,
    pub device: Device,
    pub lite: bool,
    pub overwrite: bool,
    pub output_sr: Option<u32>,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            voice: Voice::default(),
            out_dir: None,
            device: Device::Cpu,
            lite: false,
            overwrite: false,
            output_sr: None,
        }
    }
}

impl Settings {
    fn path() -> Option<PathBuf> {
        dirs::config_dir().map(|d| d.join("boin").join("config.toml"))
    }

    pub fn load() -> Settings {
        let Some(p) = Self::path() else {
            return Settings::default();
        };
        let mut s: Settings = std::fs::read_to_string(&p)
            .ok()
            .and_then(|t| toml::from_str(&t).ok())
            .unwrap_or_default();
        if !s.device.is_supported_here() {
            s.device = Device::Cpu;
        }
        s
    }

    pub fn save(&self) {
        let Some(p) = Self::path() else { return };
        if let Some(parent) = p.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(t) = toml::to_string_pretty(self) {
            let _ = std::fs::write(&p, t);
        }
    }
}
