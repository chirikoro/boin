//! Descriptive provenance only. Source values never override the validated graph
//! configuration or streaming contract. Do not serialize pickle Debug output:
//! containers can cycle, tensors are huge, and objects are not JSON metadata.
use crate::pickle::Value;
use crate::{ConvertOptions, ExportMode, ParsedCheckpoint};
use serde_json::{json, Map, Value as Json};

const FIELDS: &[&str] = &[
    "config",
    "sr",
    "f0",
    "version",
    "vocoder",
    "info",
    "description",
    "model_name",
    "name",
    "author",
    "license",
    "license_url",
    "terms",
    "epoch",
    "step",
    "creation_date",
    "dataset_length",
    "embedder_model",
    "model_hash",
    "speaker_info",
    "speakers",
    "speakers_id",
    "merge_recipe",
    "producer_name",
    "producer_version",
];

pub(crate) fn extract(root: &Value) -> (Map<String, Json>, Vec<String>) {
    let mut result = Map::new();
    let mut omitted = Vec::new();
    let mut total = 0;
    for &key in FIELDS {
        let Some(value) = root.dict_get(key) else {
            continue;
        };
        let mut budget = 8192;
        let value = to_json(&value, 0, &mut budget);
        match value {
            Some(value)
                if value.to_string().len() <= 65_536
                    && total + value.to_string().len() <= 262_144 =>
            {
                total += value.to_string().len();
                result.insert(key.into(), value);
            }
            _ => omitted.push(key.into()),
        }
    }
    (result, omitted)
}

fn to_json(value: &Value, depth: usize, budget: &mut usize) -> Option<Json> {
    if depth > 12 || *budget == 0 {
        return None;
    }
    *budget -= 1;
    Some(match value {
        Value::None => Json::Null,
        Value::Bool(v) => Json::Bool(*v),
        Value::Int(v) => json!(v),
        Value::Float(v) => Json::Number(serde_json::Number::from_f64(*v)?),
        Value::Str(v) => {
            // Never add the exporting machine's paths; reject explicit paths in
            // known descriptive fields as well. Basenames and HTTPS URLs remain.
            if v.len() > 65_536
                || v.starts_with('/')
                || v.starts_with('\\')
                || v.starts_with("file:")
                || v.as_bytes().get(1) == Some(&b':')
            {
                return None;
            }
            Json::String(v.to_string())
        }
        Value::List(items) => Json::Array(
            items
                .borrow()
                .iter()
                .map(|v| to_json(v, depth + 1, budget))
                .collect::<Option<_>>()?,
        ),
        Value::Tuple(items) => Json::Array(
            items
                .iter()
                .map(|v| to_json(v, depth + 1, budget))
                .collect::<Option<_>>()?,
        ),
        Value::Dict(items) => {
            let mut result = Map::new();
            for (key, value) in items.borrow().iter() {
                let key = match key {
                    Value::Str(v) if v.len() <= 256 => v.to_string(),
                    Value::Int(v) => v.to_string(),
                    _ => return None,
                };
                if result
                    .insert(key, to_json(value, depth + 1, budget)?)
                    .is_some()
                {
                    return None;
                }
            }
            Json::Object(result)
        }
        _ => return None,
    })
}

pub(crate) fn export(
    checkpoint: &ParsedCheckpoint,
    options: &ConvertOptions,
) -> Vec<(String, String)> {
    let mut common = json!({
        "f0": checkpoint.use_f0,
        "samplingRate": checkpoint.config.sr,
        "version": checkpoint.version.as_str(),
    });
    // Only explicitly named, valid IDs become a GUI speaker list. Embedding
    // capacity is not a count of trained voices (many single-voice PTHs use 109).
    let mut speakers = Map::new();
    if let Some(items) = checkpoint
        .source_metadata
        .get("speaker_info")
        .and_then(Json::as_array)
    {
        for item in items {
            if let (Some(id), Some(name)) = (
                item.get("id")
                    .and_then(|v| v.as_u64().or_else(|| v.as_str()?.parse().ok())),
                item.get("name").and_then(Json::as_str),
            ) {
                if id < checkpoint.config.spk_embed_dim as u64 && !name.trim().is_empty() {
                    speakers
                        .entry(id.to_string())
                        .or_insert_with(|| json!(name));
                }
            }
        }
    }
    if let Some(items) = checkpoint.source_metadata.get("speakers") {
        let pairs: Vec<_> = match items {
            Json::Object(items) => items.iter().map(|(id, name)| (id.clone(), name)).collect(),
            Json::Array(items) => items
                .iter()
                .enumerate()
                .map(|(id, name)| (id.to_string(), name))
                .collect(),
            _ => Vec::new(),
        };
        for (id, name) in pairs {
            if let (Ok(id), Some(name)) = (id.parse::<usize>(), name.as_str()) {
                if id < checkpoint.config.spk_embed_dim && !name.trim().is_empty() {
                    speakers
                        .entry(id.to_string())
                        .or_insert_with(|| json!(name));
                }
            }
        }
    }
    if !speakers.is_empty() {
        common["speakers"] = Json::Object(speakers);
    }
    vec![
        ("metadata".into(), common.to_string()),
        ("vc-rs.source".into(), json!({"schema_version": 1, "fields": checkpoint.source_metadata}).to_string()),
        ("vc-rs.export".into(), json!({
            "schema_version": 1,
            "format": match options.export_mode { ExportMode::Streaming => "streaming", ExportMode::Webui => "webui" },
            "opset": options.opset_version,
            "producer_name": "vc-convert",
            "producer_version": env!("CARGO_PKG_VERSION"),
            "speaker_capacity": checkpoint.config.spk_embed_dim,
            "omitted_source_fields": checkpoint.omitted_metadata,
        }).to_string()),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{cell::RefCell, rc::Rc};

    #[test]
    fn extracts_nested_provenance_without_serializing_cycles_or_paths() {
        let cycle = Rc::new(RefCell::new(Vec::new()));
        cycle.borrow_mut().push(Value::List(cycle.clone()));
        let root = Value::Dict(Rc::new(RefCell::new(vec![
            (Value::Str("info".into()), Value::Str("1500epoch".into())),
            (
                Value::Str("producer_name".into()),
                Value::Str("original-tool".into()),
            ),
            (Value::Str("description".into()), Value::List(cycle.clone())),
            (
                Value::Str("model_name".into()),
                Value::Str("C:\\private\\model.pth".into()),
            ),
            (Value::Str("step".into()), Value::Float(f64::NAN)),
        ])));
        let (fields, omitted) = extract(&root);
        assert_eq!(fields["info"], "1500epoch");
        assert_eq!(fields["producer_name"], "original-tool");
        assert!(omitted.contains(&"description".into()));
        assert!(omitted.contains(&"model_name".into()));
        assert!(omitted.contains(&"step".into()));
        cycle.borrow_mut().clear();
    }

    #[test]
    fn provenance_roundtrips_in_both_modes_without_changing_graph_bytes() {
        for mode in [ExportMode::Streaming, ExportMode::Webui] {
            let mut checkpoint = crate::parse_pth(crate::test_fixtures::tiny_v2_f0_pth()).unwrap();
            let options = ConvertOptions {
                export_mode: mode,
                ..Default::default()
            };
            let mut baseline =
                crate::graph::synthesizer::build_model(&checkpoint, &options).unwrap();
            checkpoint.source_metadata.insert(
                "speaker_info".into(),
                json!([
                    {"id": 2, "name": "声B"}, {"id": 0, "name": "Voice A"},
                    {"id": 2, "name": "duplicate"}, {"id": 999, "name": "invalid"},
                ]),
            );
            checkpoint.source_metadata.insert("merge_recipe".into(), json!({"sources": [{"filename": "source.pth", "weight": 0.75, "file_sha256": "abc"}]}));
            checkpoint
                .source_metadata
                .insert("producer_name".into(), json!("training-tool"));
            let mut model = crate::graph::synthesizer::build_model(&checkpoint, &options).unwrap();
            let props: Map<String, Json> = model
                .metadata_props
                .iter()
                .filter(|(k, _)| k == "metadata" || k.starts_with("vc-rs."))
                .map(|(k, v)| (k.clone(), serde_json::from_str(v).unwrap()))
                .collect();
            assert_eq!(
                props["metadata"]["speakers"],
                json!({"0": "Voice A", "2": "声B"})
            );
            assert_eq!(
                props["vc-rs.source"]["fields"]["merge_recipe"],
                checkpoint.source_metadata["merge_recipe"]
            );
            assert_eq!(
                props["vc-rs.source"]["fields"]["producer_name"],
                "training-tool"
            );
            assert_eq!(model.producer_name, "vc-convert");
            assert_eq!(props["metadata"]["version"], "v2");
            let serialized = crate::onnx::writer::serialize(&model);
            assert!(serialized
                .windows(b"vc-rs.source".len())
                .any(|v| v == b"vc-rs.source"));
            assert_eq!(serialized, crate::onnx::writer::serialize(&model));
            baseline.metadata_props.clear();
            model.metadata_props.clear();
            assert_eq!(
                crate::onnx::writer::serialize(&baseline),
                crate::onnx::writer::serialize(&model)
            );
        }
    }

    #[test]
    fn embedding_capacity_does_not_create_named_speakers() {
        let checkpoint = crate::parse_pth(crate::test_fixtures::tiny_v2_f0_pth()).unwrap();
        let props = export(&checkpoint, &ConvertOptions::default());
        let common: Json = serde_json::from_str(&props[0].1).unwrap();
        assert!(common.get("speakers").is_none());
    }

    #[test]
    #[ignore = "Reads a user-supplied PTH from VC_RS_INSPECT_PTH; no inference or file writes"]
    fn inspect_local_source_metadata() {
        let path = std::env::var_os("VC_RS_INSPECT_PTH").expect("VC_RS_INSPECT_PTH");
        let checkpoint = crate::parse_pth(&std::fs::read(path).unwrap()).unwrap();
        println!(
            "{}",
            serde_json::to_string_pretty(&checkpoint.source_metadata).unwrap()
        );
        println!("Omitted fields: {:?}", checkpoint.omitted_metadata);
        let props = export(&checkpoint, &ConvertOptions::default());
        assert!(!props.is_empty());
    }
}
