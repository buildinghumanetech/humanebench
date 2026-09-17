//! Single judge via OpenRouter.
//!
//! One call per turn, one model. The report records the model and regime so single-judge
//! numbers are never silently compared against ensemble ones.

use super::{truncate, Usage};
use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::time::Duration;

pub const DEFAULT_MODEL: &str = "anthropic/claude-sonnet-4.5";
pub const API_URL: &str = "https://openrouter.ai/api/v1/chat/completions";
/// Named on the consent screen, and the key consent is stored under.
pub const DESTINATION: &str = "OpenRouter (openrouter.ai)";
const ENV_KEY: &str = "OPENROUTER_API_KEY";

pub struct Judge {
    api_key: String,
    pub model: String,
    pub max_retries: u32,
}

impl Judge {
    pub fn from_env(model: &str) -> Result<Self> {
        let api_key = std::env::var(ENV_KEY).map_err(|_| {
            anyhow::anyhow!(
                "{ENV_KEY} is not set.\n\n\
                 Scoring is the only command that calls out to a model. Export a key:\n    \
                 export {ENV_KEY}=sk-or-...\n\n\
                 `humanebench score --dry-run` needs no key and prints the call count first."
            )
        })?;
        Ok(Judge {
            api_key,
            model: model.to_string(),
            max_retries: 3,
        })
    }

    pub fn complete(&self, prompt: &str) -> Result<(String, Usage)> {
        let body = json!({
            "model": self.model,
            // Deterministic as the API allows: this is measurement, not generation.
            "temperature": 0,
            "messages": [{ "role": "user", "content": prompt }],
        });

        let mut last_err: Option<anyhow::Error> = None;

        for attempt in 0..=self.max_retries {
            if attempt > 0 {
                // Plain exponential backoff; nothing here is latency-sensitive.
                std::thread::sleep(Duration::from_millis(500 * (1 << attempt)));
            }

            let response = ureq::post(API_URL)
                .set("Authorization", &format!("Bearer {}", self.api_key))
                .set("Content-Type", "application/json")
                .set(
                    "HTTP-Referer",
                    "https://github.com/buildinghumanetech/humanebench",
                )
                .set("X-Title", "HumaneBench CLI")
                .timeout(Duration::from_secs(180))
                .send_json(&body);

            match response {
                Ok(resp) => {
                    let v: Value = resp.into_json().context("judge response was not JSON")?;
                    return parse_completion(&v);
                }
                Err(ureq::Error::Status(code, resp)) => {
                    let detail = resp.into_string().unwrap_or_default();
                    let retryable = code == 429 || (500..600).contains(&code);
                    let err =
                        anyhow::anyhow!("OpenRouter returned {code}: {}", truncate(&detail, 400));
                    if !retryable {
                        return Err(err);
                    }
                    last_err = Some(err);
                }
                Err(e) => last_err = Some(anyhow::anyhow!("request to OpenRouter failed: {e}")),
            }
        }

        Err(last_err.unwrap_or_else(|| anyhow::anyhow!("judge call failed")))
    }
}

fn parse_completion(v: &Value) -> Result<(String, Usage)> {
    if let Some(err) = v.get("error") {
        bail!("OpenRouter error: {err}");
    }
    let content = v
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .ok_or_else(|| anyhow::anyhow!("judge response had no message content"))?;

    let usage = v
        .get("usage")
        .map(|u| Usage {
            prompt_tokens: u.get("prompt_tokens").and_then(|t| t.as_u64()).unwrap_or(0),
            completion_tokens: u
                .get("completion_tokens")
                .and_then(|t| t.as_u64())
                .unwrap_or(0),
        })
        .unwrap_or_default();

    Ok((content.to_string(), usage))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_completion_and_usage() {
        let v = serde_json::json!({
            "choices":[{"message":{"content":"{\"ok\":true}"}}],
            "usage":{"prompt_tokens":1200,"completion_tokens":300}
        });
        let (content, usage) = parse_completion(&v).unwrap();
        assert_eq!(content, "{\"ok\":true}");
        assert_eq!(usage.prompt_tokens, 1200);
        assert_eq!(usage.completion_tokens, 300);
    }

    #[test]
    fn surfaces_api_errors() {
        let v = serde_json::json!({"error":{"message":"no credits"}});
        assert!(parse_completion(&v).is_err());
    }

    #[test]
    fn missing_content_is_an_error() {
        let v = serde_json::json!({"choices":[]});
        assert!(parse_completion(&v).is_err());
    }
}
