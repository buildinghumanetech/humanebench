//! Single judge via Google Vertex AI.
//!
//! Project, location, and access token all resolve once, when the judge is built. A
//! scoring run that cannot reach Vertex should say so before the consent screen, not
//! after the first turn has already been sent.

use super::{truncate, Usage};
use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::cell::RefCell;
use std::process::Command;
use std::time::Duration;

pub const DEFAULT_MODEL: &str = "gemini-3.1-flash-lite";
// The Gemini 3.1 family is served only from `global`; the regional endpoints 404 on it.
pub const DEFAULT_LOCATION: &str = "global";

const ENV_PROJECT: &str = "GOOGLE_CLOUD_PROJECT";
const ENV_LOCATION: &str = "GOOGLE_CLOUD_LOCATION";
const ENV_TOKEN: &str = "GOOGLE_VERTEX_ACCESS_TOKEN";

const PROJECT_HINT: &str = "no Google Cloud project is set.\n\n\
     Pick one:\n    \
     gcloud config set project YOUR_PROJECT_ID\n\
     or set it for this run only:\n    \
     export GOOGLE_CLOUD_PROJECT=YOUR_PROJECT_ID";

const TOKEN_HINT: &str = "no Vertex AI access token.\n\n\
     Log in:\n    \
     gcloud auth login\n\
     or supply a token directly:\n    \
     export GOOGLE_VERTEX_ACCESS_TOKEN=$(gcloud auth print-access-token)";

pub struct Judge {
    project: String,
    location: String,
    /// Replaced in place after a 401: a gcloud token lasts about an hour, which a long
    /// scoring run outlives.
    access_token: RefCell<String>,
    pub model: String,
    pub max_retries: u32,
}

impl Judge {
    pub fn from_env(model: &str) -> Result<Self> {
        Ok(Judge {
            project: resolve_project()?,
            location: env_value(ENV_LOCATION).unwrap_or_else(|| DEFAULT_LOCATION.to_string()),
            access_token: RefCell::new(resolve_access_token()?),
            model: model.to_string(),
            max_retries: 3,
        })
    }

    /// Named on the consent screen. The project id is the part that matters: sending
    /// personal conversation history to a company project is a different decision.
    pub fn destination(&self) -> String {
        format!(
            "Google Vertex AI (project {}, location {})",
            self.project, self.location
        )
    }

    pub fn complete(&self, prompt: &str) -> Result<(String, Usage)> {
        let body = json!({
            "contents": [{ "role": "user", "parts": [{ "text": prompt }] }],
            // Deterministic as the API allows: this is measurement, not generation.
            "generationConfig": { "temperature": 0, "responseMimeType": "application/json" },
        });
        let url = endpoint(&self.location, &self.project, &self.model);

        let mut last_err: Option<anyhow::Error> = None;
        let mut refreshed = false;

        for attempt in 0..=self.max_retries {
            if attempt > 0 {
                // Plain exponential backoff; nothing here is latency-sensitive.
                std::thread::sleep(Duration::from_millis(500 * (1 << attempt)));
            }

            let token = self.access_token.borrow().clone();
            let response = ureq::post(&url)
                .set("Authorization", &format!("Bearer {token}"))
                .set("Content-Type", "application/json")
                .timeout(Duration::from_secs(180))
                .send_json(&body);

            match response {
                Ok(resp) => {
                    let v: Value = resp.into_json().context("judge response was not JSON")?;
                    return parse_completion(&v);
                }
                Err(ureq::Error::Status(401, _)) if !refreshed => {
                    refreshed = true;
                    *self.access_token.borrow_mut() = resolve_access_token()?;
                    last_err = Some(anyhow::anyhow!(
                        "Vertex returned 401; refreshed the access token"
                    ));
                }
                Err(ureq::Error::Status(code, resp)) => {
                    let detail = resp.into_string().unwrap_or_default();
                    let retryable = code == 429 || (500..600).contains(&code);
                    let err = anyhow::anyhow!("Vertex returned {code}: {}", truncate(&detail, 400));
                    if !retryable {
                        return Err(err);
                    }
                    last_err = Some(err);
                }
                Err(e) => last_err = Some(anyhow::anyhow!("request to Vertex failed: {e}")),
            }
        }

        Err(last_err.unwrap_or_else(|| anyhow::anyhow!("judge call failed")))
    }
}

/// Regional endpoints are hosted per location; `global` is the one that has no prefix.
fn endpoint(location: &str, project: &str, model: &str) -> String {
    let host = if location == "global" {
        "https://aiplatform.googleapis.com".to_string()
    } else {
        format!("https://{location}-aiplatform.googleapis.com")
    };
    format!(
        "{host}/v1/projects/{project}/locations/{location}/publishers/google/models/{model}:generateContent"
    )
}

fn env_value(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|v| !v.trim().is_empty())
}

/// Run gcloud and return its trimmed stdout.
fn gcloud(args: &[&str]) -> Result<String> {
    let out = Command::new("gcloud")
        .args(args)
        .output()
        .with_context(|| format!("running `gcloud {}`", args.join(" ")))?;
    if !out.status.success() {
        bail!(
            "`gcloud {}` failed: {}",
            args.join(" "),
            truncate(String::from_utf8_lossy(&out.stderr).trim(), 400)
        );
    }
    let value = String::from_utf8_lossy(&out.stdout).trim().to_string();
    // gcloud prints this rather than an empty line when a property has no value.
    Ok(if value == "(unset)" {
        String::new()
    } else {
        value
    })
}

fn resolve_project() -> Result<String> {
    if let Some(project) = env_value(ENV_PROJECT) {
        return Ok(project);
    }
    let project = gcloud(&["config", "get-value", "project"]).context(PROJECT_HINT)?;
    if project.is_empty() {
        bail!("{PROJECT_HINT}");
    }
    Ok(project)
}

fn resolve_access_token() -> Result<String> {
    if let Some(token) = env_value(ENV_TOKEN) {
        return Ok(token);
    }
    let token = gcloud(&["auth", "print-access-token"]).context(TOKEN_HINT)?;
    if token.is_empty() {
        bail!("{TOKEN_HINT}");
    }
    Ok(token)
}

fn parse_completion(v: &Value) -> Result<(String, Usage)> {
    if let Some(err) = v.get("error") {
        let field = |name: &str| err.get(name).and_then(Value::as_str).unwrap_or("");
        bail!(
            "Vertex error {} {}: {}",
            err.get("code").and_then(Value::as_i64).unwrap_or(0),
            field("status"),
            truncate(field("message"), 400)
        );
    }

    let counts = v.get("usageMetadata");
    let count = |name: &str| {
        counts
            .and_then(|u| u.get(name))
            .and_then(Value::as_u64)
            .unwrap_or(0)
    };
    let thinking = count("thoughtsTokenCount");
    let usage = Usage {
        prompt_tokens: count("promptTokenCount"),
        // Thinking tokens bill as output.
        completion_tokens: count("candidatesTokenCount") + thinking,
    };

    let candidate = v
        .get("candidates")
        .and_then(|c| c.get(0))
        .ok_or_else(|| anyhow::anyhow!("judge response had no candidates"))?;

    // A truncated answer leaves only thought parts behind, which would otherwise reach
    // the caller as an unexplained empty response and repeat on every turn.
    if let Some(reason) = candidate.get("finishReason").and_then(Value::as_str) {
        if reason != "STOP" {
            bail!(
                "Vertex stopped early: finishReason {reason} \
                 ({} prompt, {} answer, {thinking} thinking tokens)",
                usage.prompt_tokens,
                count("candidatesTokenCount")
            );
        }
    }

    let text: String = candidate
        .get("content")
        .and_then(|c| c.get("parts"))
        .and_then(Value::as_array)
        .map(|parts| {
            parts
                .iter()
                .filter(|p| !p.get("thought").and_then(Value::as_bool).unwrap_or(false))
                .filter_map(|p| p.get("text").and_then(Value::as_str))
                .collect()
        })
        .unwrap_or_default();

    if text.trim().is_empty() {
        bail!("judge response had no answer text");
    }
    Ok((text, usage))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regional_and_global_endpoints_use_different_hosts() {
        assert_eq!(
            endpoint("us-central1", "p", "gemini-3.1-pro-preview"),
            "https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/us-central1/publishers/google/models/gemini-3.1-pro-preview:generateContent"
        );
        assert_eq!(
            endpoint("global", "p", "gemini-3.1-pro-preview"),
            "https://aiplatform.googleapis.com/v1/projects/p/locations/global/publishers/google/models/gemini-3.1-pro-preview:generateContent"
        );
    }

    #[test]
    fn skips_thought_parts_and_bills_thinking_as_output() {
        let v = serde_json::json!({
            "candidates": [{
                "content": { "role": "model", "parts": [
                    { "text": "weighing the rubric", "thought": true },
                    { "text": "{\"ok\":" },
                    { "text": "true}" }
                ]},
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 1200,
                "candidatesTokenCount": 300,
                "thoughtsTokenCount": 900
            }
        });
        let (content, usage) = parse_completion(&v).unwrap();
        assert_eq!(content, "{\"ok\":true}");
        assert_eq!(usage.prompt_tokens, 1200);
        assert_eq!(usage.completion_tokens, 1200);
    }

    #[test]
    fn truncation_names_the_finish_reason() {
        let v = serde_json::json!({
            "candidates": [{
                "content": { "role": "model", "parts": [
                    { "text": "still weighing", "thought": true }
                ]},
                "finishReason": "MAX_TOKENS"
            }],
            "usageMetadata": {
                "promptTokenCount": 1200,
                "candidatesTokenCount": 0,
                "thoughtsTokenCount": 65000
            }
        });
        let err = parse_completion(&v).unwrap_err().to_string();
        assert!(err.contains("MAX_TOKENS"), "got: {err}");
        assert!(err.contains("65000"), "got: {err}");
    }

    #[test]
    fn surfaces_api_errors() {
        let v = serde_json::json!({"error": {
            "code": 403,
            "message": "Permission denied on resource project foo.",
            "status": "PERMISSION_DENIED"
        }});
        let err = parse_completion(&v).unwrap_err().to_string();
        assert!(err.contains("403"), "got: {err}");
        assert!(err.contains("PERMISSION_DENIED"), "got: {err}");
    }

    #[test]
    fn missing_content_is_an_error() {
        let v = serde_json::json!({"candidates": []});
        assert!(parse_completion(&v).is_err());
    }
}
