//! boin GUI: 音声ファイルをドラッグ&ドロップで一括変換するデスクトップアプリ。

#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

mod settings;
mod worker;

use boin_core::audio::INPUT_EXTENSIONS;
use boin_core::{prepare, Device, Home, Voice};
use eframe::egui;
use settings::Settings;
use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use worker::{FileStatus, Job, Msg};

fn main() -> eframe::Result {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("warn".parse().unwrap()),
        )
        .with_target(false)
        .init();
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("boin - 音声変換")
            .with_inner_size([860.0, 640.0])
            .with_min_inner_size([640.0, 480.0])
            .with_drag_and_drop(true),
        ..Default::default()
    };
    eframe::run_native(
        "boin",
        options,
        Box::new(|cc| {
            install_fonts(&cc.egui_ctx);
            Ok(Box::new(App::new()))
        }),
    )
}

/// 日本語表示用に BIZ UDPGothic（SIL Open Font License 1.1）を同梱して登録する。
fn install_fonts(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert(
        "biz_udp_gothic".to_owned(),
        Arc::new(egui::FontData::from_static(include_bytes!(
            "../assets/BIZUDPGothic-Regular.ttf"
        ))),
    );
    for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
        fonts
            .families
            .entry(family)
            .or_default()
            .insert(0, "biz_udp_gothic".to_owned());
    }
    ctx.set_fonts(fonts);
    ctx.all_styles_mut(|s| {
        for font in s.text_styles.values_mut() {
            font.size *= 1.1;
        }
    });
}

struct FileEntry {
    path: PathBuf,
    status: FileStatus,
}

struct App {
    home: Home,
    /// 起動時の致命的エラー（onnxruntime.dll が無い等）。
    fatal: Option<String>,
    settings: Settings,
    files: Vec<FileEntry>,
    running: bool,
    cancel: Arc<AtomicBool>,
    rx: Option<Receiver<Msg>>,
    progress_text: String,
    progress_frac: Option<f32>,
    log: Vec<String>,
    ready: bool,
    status_line: String,
}

impl App {
    fn new() -> App {
        let home = Home::detect();
        let settings = Settings::load();
        let ready = prepare::is_ready(&home, settings.voice, settings.lite);
        let fatal = boin_core::onnx::init_runtime(&home)
            .err()
            .map(|e| format!("{e:#}"));
        App {
            home,
            fatal,
            settings,
            files: Vec::new(),
            running: false,
            cancel: Arc::new(AtomicBool::new(false)),
            rx: None,
            progress_text: String::new(),
            progress_frac: None,
            log: Vec::new(),
            ready,
            status_line: String::new(),
        }
    }

    fn add_paths(&mut self, paths: Vec<PathBuf>) {
        for p in paths {
            if p.is_dir() {
                if let Ok(rd) = std::fs::read_dir(&p) {
                    let mut children: Vec<PathBuf> = rd.flatten().map(|e| e.path()).collect();
                    children.sort();
                    for c in children {
                        if c.is_file() && is_audio(&c) {
                            self.push_file(c);
                        }
                    }
                }
            } else if p.is_file() {
                self.push_file(p);
            }
        }
    }

    fn push_file(&mut self, p: PathBuf) {
        if self.files.iter().any(|f| f.path == p) {
            return;
        }
        self.files.push(FileEntry {
            path: p,
            status: FileStatus::Pending,
        });
    }

    fn output_dir_for(&self, input: &Path) -> PathBuf {
        match &self.settings.out_dir {
            Some(d) if !d.as_os_str().is_empty() => d.clone(),
            _ => input
                .parent()
                .map(|p| p.to_path_buf())
                .unwrap_or_else(|| PathBuf::from(".")),
        }
    }

    fn start_convert(&mut self) {
        let jobs: Vec<Job> = self
            .files
            .iter()
            .enumerate()
            .filter(|(_, f)| !matches!(f.status, FileStatus::Done(_)))
            .map(|(i, f)| Job {
                index: i,
                input: f.path.clone(),
                out_dir: self.output_dir_for(&f.path),
            })
            .collect();
        if jobs.is_empty() {
            self.status_line = "変換するファイルがありません".to_owned();
            return;
        }
        for j in &jobs {
            self.files[j.index].status = FileStatus::Pending;
        }
        self.cancel.store(false, Ordering::SeqCst);
        let rx = worker::spawn_convert(
            self.home.clone(),
            self.settings.clone(),
            jobs,
            Arc::clone(&self.cancel),
        );
        self.rx = Some(rx);
        self.running = true;
        self.progress_text = "モデルを読み込み中…".to_owned();
        self.progress_frac = None;
    }

    fn start_prepare(&mut self) {
        let rx = worker::spawn_prepare(self.home.clone(), self.settings.clone());
        self.rx = Some(rx);
        self.running = true;
        self.progress_text = "モデルを準備中…".to_owned();
        self.progress_frac = None;
    }

    fn poll_worker(&mut self) {
        let Some(rx) = &self.rx else { return };
        let mut finished = false;
        while let Ok(msg) = rx.try_recv() {
            match msg {
                Msg::Log(s) => self.log.push(s),
                Msg::Progress { text, frac } => {
                    self.progress_text = text;
                    self.progress_frac = frac;
                }
                Msg::File { index, status } => {
                    if let Some(f) = self.files.get_mut(index) {
                        f.status = status;
                    }
                }
                Msg::Finished { summary } => {
                    self.status_line = summary;
                    finished = true;
                }
            }
        }
        if finished {
            self.rx = None;
            self.running = false;
            self.progress_frac = None;
            self.progress_text.clear();
            self.ready = prepare::is_ready(&self.home, self.settings.voice, self.settings.lite);
        }
    }
}

fn is_audio(p: &Path) -> bool {
    p.extension()
        .and_then(|e| e.to_str())
        .map(|e| INPUT_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        let ctx = &ctx;
        self.poll_worker();
        if self.running {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        } else {
            // 設定（声モデル / 軽量モード）の変更に応じて準備状況を更新する
            self.ready = prepare::is_ready(&self.home, self.settings.voice, self.settings.lite);
        }

        // ドラッグ&ドロップ
        let dropped: Vec<PathBuf> = ctx.input(|i| {
            i.raw
                .dropped_files
                .iter()
                .map(|f| f.path().to_path_buf())
                .collect()
        });
        if !dropped.is_empty() && !self.running {
            self.add_paths(dropped);
        }
        let hovering = ctx.input(|i| !i.raw.hovered_files.is_empty());

        egui::Panel::top("top").show(ui, |ui| {
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                ui.heading("boin 音声変換");
                ui.label(egui::RichText::new("愛想良い系少女の声 V2").weak());
            });
            if let Some(err) = &self.fatal {
                ui.label(egui::RichText::new(err).color(egui::Color32::from_rgb(200, 40, 40)));
            }
            ui.add_space(4.0);
        });

        egui::Panel::left("settings").default_size(270.0).min_size(250.0).show(ui, |ui| {
            ui.add_space(8.0);
            ui.add_enabled_ui(!self.running, |ui| {
                ui.label("声モデル");
                egui::ComboBox::from_id_salt("voice")
                    .width(220.0)
                    .selected_text(self.settings.voice.display_ja())
                    .show_ui(ui, |ui| {
                        for v in Voice::ALL {
                            ui.selectable_value(&mut self.settings.voice, v, v.display_ja());
                        }
                    });
                ui.add_space(8.0);

                ui.label("出力フォルダ");
                let out_text = self
                    .settings
                    .out_dir
                    .as_ref()
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|| "（入力と同じ場所）".to_owned());
                ui.label(egui::RichText::new(out_text).small()).on_hover_text("変換結果の保存先");
                ui.horizontal(|ui| {
                    if ui.button("選択…").clicked() {
                        if let Some(d) = rfd::FileDialog::new().pick_folder() {
                            self.settings.out_dir = Some(d);
                        }
                    }
                    if ui.button("入力と同じ").clicked() {
                        self.settings.out_dir = None;
                    }
                });
                ui.add_space(8.0);

                ui.label("処理デバイス");
                egui::ComboBox::from_id_salt("device")
                    .width(220.0)
                    .selected_text(self.settings.device.display_ja())
                    .show_ui(ui, |ui| {
                        for d in Device::available() {
                            ui.selectable_value(&mut self.settings.device, d, d.display_ja());
                        }
                    });
                ui.add_space(8.0);

                ui.checkbox(&mut self.settings.overwrite, "既存の出力を上書き");
                ui.checkbox(&mut self.settings.lite, "軽量モード（int8 ContentVec）")
                    .on_hover_text("特徴抽出モデルを 1/4 サイズの量子化版にします。やや音質が落ちる場合があります。");
                ui.add_space(8.0);
                ui.label("出力サンプルレート");
                egui::ComboBox::from_id_salt("sr")
                    .width(220.0)
                    .selected_text(match self.settings.output_sr {
                        None => "40000 Hz（モデル標準）".to_owned(),
                        Some(sr) => format!("{sr} Hz"),
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.settings.output_sr, None, "40000 Hz（モデル標準）");
                        ui.selectable_value(&mut self.settings.output_sr, Some(44100), "44100 Hz");
                        ui.selectable_value(&mut self.settings.output_sr, Some(48000), "48000 Hz");
                        ui.selectable_value(&mut self.settings.output_sr, Some(16000), "16000 Hz");
                    });
            });

            ui.add_space(16.0);
            ui.separator();
            ui.add_space(8.0);
            if self.ready {
                ui.label(egui::RichText::new("✓ モデル準備済み").color(egui::Color32::from_rgb(40, 140, 60)));
            } else {
                ui.label(egui::RichText::new("モデルが未準備です").color(egui::Color32::from_rgb(200, 120, 0)));
                ui.label(egui::RichText::new("初回は基盤モデル（約 380MB）のダウンロードと声モデルの変換を行います。").small());
                if ui.add_enabled(!self.running && self.fatal.is_none(), egui::Button::new("モデルを準備")).clicked() {
                    self.start_prepare();
                }
            }
            ui.add_space(8.0);
            ui.label(egui::RichText::new(format!("フォルダ: {}", self.home.root.display())).small().weak());
        });

        egui::Panel::bottom("bottom").show(ui, |ui| {
            ui.add_space(6.0);
            if self.running {
                let bar = match self.progress_frac {
                    Some(f) => egui::ProgressBar::new(f).text(self.progress_text.clone()),
                    None => egui::ProgressBar::new(0.0)
                        .animate(true)
                        .text(self.progress_text.clone()),
                };
                ui.add(bar);
            } else if !self.status_line.is_empty() {
                ui.label(&self.status_line);
            }
            ui.add_space(4.0);
            egui::CollapsingHeader::new("ログ")
                .default_open(false)
                .show(ui, |ui| {
                    egui::ScrollArea::vertical()
                        .max_height(140.0)
                        .stick_to_bottom(true)
                        .show(ui, |ui| {
                            for line in &self.log {
                                ui.label(egui::RichText::new(line).monospace().small());
                            }
                        });
                });
            ui.add_space(4.0);
        });

        egui::CentralPanel::default().show(ui, |ui| {
            ui.horizontal(|ui| {
                if ui
                    .add_enabled(!self.running, egui::Button::new("ファイルを追加…"))
                    .clicked()
                {
                    if let Some(files) = rfd::FileDialog::new()
                        .add_filter("音声ファイル", INPUT_EXTENSIONS)
                        .pick_files()
                    {
                        self.add_paths(files);
                    }
                }
                if ui
                    .add_enabled(!self.running, egui::Button::new("フォルダを追加…"))
                    .clicked()
                {
                    if let Some(d) = rfd::FileDialog::new().pick_folder() {
                        self.add_paths(vec![d]);
                    }
                }
                if ui
                    .add_enabled(
                        !self.running && !self.files.is_empty(),
                        egui::Button::new("一覧をクリア"),
                    )
                    .clicked()
                {
                    self.files.clear();
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if self.running {
                        if ui.button("中止").clicked() {
                            self.cancel.store(true, Ordering::SeqCst);
                        }
                    } else {
                        let can_run = !self.files.is_empty() && self.fatal.is_none();
                        let btn = egui::Button::new(egui::RichText::new("変換開始").strong());
                        if ui.add_enabled(can_run, btn).clicked() {
                            self.settings.save();
                            self.start_convert();
                        }
                    }
                });
            });
            ui.add_space(6.0);

            let frame = egui::Frame::group(ui.style()).fill(if hovering {
                ui.visuals().selection.bg_fill.gamma_multiply(0.3)
            } else {
                ui.visuals().faint_bg_color
            });
            frame.show(ui, |ui| {
                ui.set_min_height(ui.available_height() - 4.0);
                ui.set_width(ui.available_width());
                if self.files.is_empty() {
                    ui.vertical_centered(|ui| {
                        ui.add_space(ui.available_height() * 0.35);
                        ui.label(
                            egui::RichText::new("ここに音声ファイルをドロップ")
                                .size(20.0)
                                .weak(),
                        );
                        ui.label(
                            egui::RichText::new(
                                "wav / mp3 / flac / ogg / m4a などに対応。フォルダも可。",
                            )
                            .weak(),
                        );
                    });
                } else {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        let mut remove: Option<usize> = None;
                        egui::Grid::new("files")
                            .striped(true)
                            .num_columns(3)
                            .min_col_width(60.0)
                            .show(ui, |ui| {
                                ui.label(egui::RichText::new("ファイル").strong());
                                ui.label(egui::RichText::new("状態").strong());
                                ui.label("");
                                ui.end_row();
                                for (i, f) in self.files.iter().enumerate() {
                                    let name =
                                        f.path.file_name().and_then(|s| s.to_str()).unwrap_or("?");
                                    ui.label(name).on_hover_text(f.path.display().to_string());
                                    match &f.status {
                                        FileStatus::Pending => {
                                            ui.label(egui::RichText::new("待機").weak());
                                        }
                                        FileStatus::Running(s) => {
                                            ui.label(
                                                egui::RichText::new(s)
                                                    .color(egui::Color32::from_rgb(30, 110, 200)),
                                            );
                                        }
                                        FileStatus::Done(s) => {
                                            ui.label(
                                                egui::RichText::new(s)
                                                    .color(egui::Color32::from_rgb(40, 140, 60)),
                                            );
                                        }
                                        FileStatus::Skipped(s) => {
                                            ui.label(
                                                egui::RichText::new(s)
                                                    .color(egui::Color32::from_rgb(200, 120, 0)),
                                            );
                                        }
                                        FileStatus::Failed(s) => {
                                            ui.label(
                                                egui::RichText::new(s)
                                                    .color(egui::Color32::from_rgb(200, 40, 40)),
                                            );
                                        }
                                    }
                                    if !self.running && ui.small_button("✕").clicked() {
                                        remove = Some(i);
                                    }
                                    ui.end_row();
                                }
                            });
                        if let Some(i) = remove {
                            self.files.remove(i);
                        }
                    });
                }
            });
        });
    }

    fn save(&mut self, _storage: &mut dyn eframe::Storage) {
        self.settings.save();
    }
}
