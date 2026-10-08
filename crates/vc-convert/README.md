# RVC conversion metadata

Both export modes retain bounded descriptive PTH fields alongside the generated
ONNX graph. The graph's validated configuration is authoritative; source metadata
cannot override the sample rate, F0 contract, or streaming inputs.

| ONNX field | Meaning |
| --- | --- |
| `producer_name`, `producer_version` | `vc-convert` and its version |
| `metadata` | Common JSON: `f0`, `samplingRate`, `version`, optional `speakers` |
| `vc-rs.source` | JSON `{schema_version: 1, fields: {...}}`, preserving source values |
| `vc-rs.export` | JSON schema version, export format, opset, converter identity, speaker capacity, omitted source fields |
| `rvc.*` | Existing streaming export contract, present only in streaming mode |

`metadata.rs::FIELDS` defines the retained source fields: original configuration,
sample-rate notation, descriptive/training information, speaker information,
source tool identity, and merge recipes including source basenames and hashes.
Absent values are never fabricated. Null and empty values can be retained without
being displayed. Unknown top-level fields, weights, training state, and executable
objects are not copied into descriptive metadata.

Each selected field is limited to 64 KiB serialized JSON, 8192 visited values,
and depth 12; the total value budget is 256 KiB. Unsupported values, cycles,
non-finite numbers, and explicit absolute-path strings omit their containing
top-level field. `omitted_source_fields` records those field names. No machine
identity, current timestamp, or input file path is added by the exporter. Source
free text is descriptive content, not a guarantee that a model is fit to publish.

Named speakers normalize from `speaker_info` entries (`id`, `name`), then
`speakers` dictionaries or arrays. IDs must be within embedding capacity; the
first valid name wins on duplicates. IDs are not renumbered. The original source
representation is retained separately. Embedding rows describe ID capacity, not
the number of trained or named voices; capacity never creates a speaker list.

The GUI reads schema version 1 and displays selected descriptive fields. Missing
speaker names hide the speaker section. Merge recipes and structural configuration
remain in the ONNX metadata without expanding the normal GUI into a graph viewer.
Existing ONNX files continue to load; recovering discarded source information
requires re-export from PTH.

Tests cover both export modes, deterministic serialization, unchanged graph and
weight bytes when provenance changes, optional speakers, and bounded parsing.
The ignored `inspect_local_source_metadata` test accepts `VC_RS_INSPECT_PTH` for
read-only inspection of a local model; it does not write or replace model files.
