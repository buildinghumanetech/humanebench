//! Hermes.
//!
//! The Hermes transcript format is an open question in the spec: whether it is a harness
//! with local logs or requires an export path determines whether it should be a compiled
//! adapter at all. Rather than ship a guess under a real source name — a wrong field name
//! in shipped code is worse than an acknowledged gap — this points at the documented
//! converter path, which needs no Rust and no merged PR.

use crate::transcript::Record;
use anyhow::{bail, Result};

pub const SOURCE: &str = "hermes";

pub fn parse(_input: &str) -> Result<Vec<Record>> {
    bail!(
        "the Hermes transcript format is not pinned down yet, so this binary ships no \
         compiled Hermes adapter.\n\n\
         Use the documented schema instead — write a converter in any language that emits \
         humanebench.transcript/v1 JSONL and pipe it in:\n\n    \
         your-hermes-converter < logs | humanebench ingest --stdin --source hermes\n\n\
         Run `humanebench schema` to print the field-by-field contract."
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn points_at_the_converter_path_instead_of_guessing() {
        let err = parse("{}").unwrap_err().to_string();
        assert!(err.contains("--stdin"));
        assert!(err.contains("humanebench.transcript/v1"));
    }
}
