//! Pure-Rust HTTP client for the local Ollama daemon (reqwest + rustls). Plain HTTP to localhost.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use base64::Engine;
use futures_util::StreamExt;
use serde::Deserialize;

use super::types::{AttributionConfig, OllamaModel};

fn http() -> reqwest::Client {
    reqwest::Client::new()
}

fn base(url: &str) -> &str {
    url.trim_end_matches('/')
}

/// Reachability heartbeat — returns the daemon version on success.
pub async fn version(base_url: &str) -> Result<String, String> {
    #[derive(Deserialize)]
    struct V {
        version: String,
    }
    let url = format!("{}/api/version", base(base_url));
    let resp = http()
        .get(&url)
        .timeout(Duration::from_secs(3))
        .send()
        .await
        .map_err(|e| e.to_string())?
        .error_for_status()
        .map_err(|e| e.to_string())?;
    let v: V = resp.json().await.map_err(|e| e.to_string())?;
    Ok(v.version)
}

/// Locally-installed models (`GET /api/tags`).
pub async fn list_tags(base_url: &str) -> Result<Vec<OllamaModel>, String> {
    #[derive(Deserialize)]
    struct Tags {
        models: Vec<Tag>,
    }
    #[derive(Deserialize)]
    struct Tag {
        name: String,
        #[serde(default)]
        size: u64,
    }
    let url = format!("{}/api/tags", base(base_url));
    let resp = http()
        .get(&url)
        .timeout(Duration::from_secs(10))
        .send()
        .await
        .map_err(|e| e.to_string())?
        .error_for_status()
        .map_err(|e| e.to_string())?;
    let t: Tags = resp.json().await.map_err(|e| e.to_string())?;
    Ok(t.models.into_iter().map(|m| OllamaModel { name: m.name, size: m.size }).collect())
}

/// One NDJSON progress line from `/api/pull`.
#[derive(Deserialize)]
pub struct PullLine {
    pub status: String,
    #[serde(default)]
    pub digest: Option<String>,
    #[serde(default)]
    pub total: Option<u64>,
    #[serde(default)]
    pub completed: Option<u64>,
}

/// Pull (download) a model, invoking `on_line` per NDJSON line. Stops early (Ok) if `cancel` is set.
pub async fn pull(
    base_url: &str,
    model: &str,
    cancel: &Arc<AtomicBool>,
    mut on_line: impl FnMut(PullLine),
) -> Result<(), String> {
    #[derive(serde::Serialize)]
    struct Req<'a> {
        model: &'a str,
        stream: bool,
    }
    let url = format!("{}/api/pull", base(base_url));
    let resp = http()
        .post(&url)
        .json(&Req { model, stream: true })
        .send()
        .await
        .map_err(|e| e.to_string())?
        .error_for_status()
        .map_err(|e| e.to_string())?;

    let mut stream = resp.bytes_stream();
    let mut buf: Vec<u8> = Vec::new();
    while let Some(chunk) = stream.next().await {
        if cancel.load(Ordering::Relaxed) {
            return Ok(()); // cooperative cancel; partial pulls are resumable
        }
        let chunk = chunk.map_err(|e| e.to_string())?;
        buf.extend_from_slice(&chunk);
        while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = buf.drain(..=pos).collect();
            let trimmed = &line[..line.len().saturating_sub(1)];
            if trimmed.is_empty() {
                continue;
            }
            if let Ok(parsed) = serde_json::from_slice::<PullLine>(trimmed) {
                on_line(parsed);
            }
        }
    }
    Ok(())
}

fn clean_folder_context(raw: &str) -> String {
    raw.replace('_', " ").replace('-', " ").split_whitespace().collect::<Vec<_>>().join(" ")
}

fn clean_filename_context(raw: &str) -> String {
    let mut parts: Vec<&str> = raw.split('_').filter(|p| !p.is_empty()).collect();

    // Generated stock filenames commonly end with a timestamp and a resolution marker,
    // e.g. Caregiver_sitting_with_happy_dogs_2K_20260916095937.
    if parts.last().is_some_and(|p| {
        p.len() >= 8 && p.len() <= 20 && p.chars().all(|c| c.is_ascii_digit())
    }) {
        parts.pop();
    }
    if parts.last().is_some_and(|p| {
        matches!(p.to_ascii_lowercase().as_str(), "2k" | "4k" | "8k" | "16k")
    }) {
        parts.pop();
    }

    parts.join(" ").split_whitespace().collect::<Vec<_>>().join(" ")
}

/// One non-streaming vision inference. Returns the model's `response` string (the strict JSON text).
pub async fn generate(cfg: &AttributionConfig, image_b64: String, image_path: &str) -> Result<String, String> {
    // Add series context without exposing the user's full filesystem path.
    // Folder and filename describe the intended commercial series and should be used actively
    // whenever they are compatible with the visible image.
    let path = std::path::Path::new(image_path);
    let folder_raw = path
        .parent()
        .and_then(|p| p.file_name())
        .and_then(|s| s.to_str())
        .unwrap_or("");
    let filename_raw = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("");

    let folder = clean_folder_context(folder_raw);
    let filename = clean_filename_context(filename_raw);

    let prompt = if folder.is_empty() && filename.is_empty() {
        cfg.prompt.clone()
    } else {
        format!(
            "{}\n\nSERIES CONTEXT:\nFolder: {}\nFilename: {}\n\nThe folder name and filename describe the intended commercial series and are trusted contextual metadata. If the series context is compatible with what is visible in the image, actively use it when creating the title, description, and keywords. Prefer specific commercial concepts from the series context over generic visual descriptions when the image does not contradict them. Use the series context especially when selecting the first 10 keywords. When the series context clearly identifies the commercial scenario, prioritize that scenario over incidental decor, furniture, flooring, shelves, room style, or other background details. Do not spend keyword slots on minor environment details unless buyers would realistically search for them. Avoid contradictory framing: for example, do not describe a scene as a home or living room when the trusted series context indicates a daycare, clinic, workshop, office, facility, or other professional setting, unless the image clearly shows a private home. Prefer standard buyer search phrases and natural nouns such as pet caregiver, dog boarding service, canine socialization, animal care professional, group of dogs. Avoid awkward generated phrases such as pet caregiver sitting, happy dogs group, toys shelf, or similar literal combinations that are unlikely buyer searches. Do not invent unsupported object functions: for example, if a bowl is visible but its contents or purpose are unclear, prefer pet bowl over cat food bowl or cat water bowl. Avoid weak marketing-style or vague phrases such as comfort zone, wellness center, premium lifestyle, cozy atmosphere, modern living, or similar filler unless they describe a clearly visible and commercially searchable concept. PRIMARY SUBJECT OVERRIDES SERIES CONTEXT: first identify what the image is mainly selling visually. The series context explains intended use, but it must not turn a product-only image into a service scene or a background into the subject. If the main visible subject is a product or accessory such as a collar, leash, bowl, toy, pet accessory, tool, device, or other standalone object, describe and keyword that object first. Use the series context only to add accurate usage context such as pet boarding or daycare. Do not infer pet grooming, pet socialization, dog owner, pet store, animal shelter, training, or other services unless those activities or settings are actually supported by the image. For product/object images, do not spend keyword slots on incidental decor such as wooden counter, modern interior, clean room, indoor plant, large window, wooden furniture, floor, wall, shelf, or similar background details unless that background is itself commercially important. CATEGORY PRIORITY: choose categories from the PRIMARY VISIBLE COMMERCIAL SUBJECT, not from incidental architecture or from the series folder alone. If a standalone product/accessory is the main visible subject, prefer Objects. If animals themselves are the main visible subject, prefer Animals/Wildlife. If an animal-care service with people/animals is the main subject, prefer Animals/Wildlife and optionally People when appropriate. If a medical procedure is the main subject, prefer Healthcare/Medical. If repair, construction, machinery, or a trade service is the main subject, prefer Industrial. Use Interiors only when the interior space itself is the primary stock subject. When two categories are allowed, put the primary commercial subject first and use the environment only as a secondary category when genuinely useful. Do not use a contextual concept only when the image clearly contradicts it. Do not infer geographic location such as USA solely from the folder name unless location metadata is explicitly required.",
            cfg.prompt, folder, filename
        )
    };
    let prompt_chars = prompt.chars().count();

    log::info!(
        "ollama series context → folder={} filename={}",
        folder,
        filename
    );

    let mut body = serde_json::json!({
        "model": cfg.model,
        "prompt": prompt,
        "images": [image_b64],
        "stream": false,
        "format": cfg.format,
    });
    let obj = body.as_object_mut().ok_or("internal: request body is not an object")?;
    if !cfg.options.is_null() {
        obj.insert("options".into(), cfg.options.clone());
    }
    if let Some(think) = &cfg.think {
        obj.insert("think".into(), serde_json::to_value(think).map_err(|e| e.to_string())?);
    }
    if let Some(keep_alive) = &cfg.keep_alive {
        obj.insert("keep_alive".into(), serde_json::Value::String(keep_alive.clone()));
    }

    // Log the effective run parameters so an operator can confirm which options actually reached the
    // model. An empty `options` here means no profile matched the active model (and no `base` profile
    // exists), so Ollama silently falls back to its own defaults instead of the configured values.
    let format_state = match cfg.format.as_object() {
        Some(o) if !o.is_empty() => "set",
        _ => "none",
    };
    log::info!(
        "ollama generate → model={} options={} think={:?} keepAlive={:?} format={} promptChars={}",
        cfg.model,
        cfg.options,
        cfg.think,
        cfg.keep_alive,
        format_state,
        prompt_chars
    );

    #[derive(Deserialize)]
    struct Gen {
        response: String,
    }
    let url = format!("{}/api/generate", base(&cfg.base_url));
    let resp = http()
        .post(&url)
        .json(&body)
        .timeout(Duration::from_secs(600))
        .send()
        .await
        .map_err(|e| format!("Ollama not reachable: {e}"))?;
    let status = resp.status();
    if !status.is_success() {
        // Surface Ollama's real error body (e.g. out-of-memory, model not found) instead of a
        // misleading downstream "invalid JSON" parse failure.
        let detail = resp.text().await.unwrap_or_default();
        return Err(format!("Ollama returned {status}: {}", detail.trim()));
    }
    let g: Gen = resp.json().await.map_err(|e| e.to_string())?;
    log::info!("ollama generate ← model={} responseChars={}", cfg.model, g.response.chars().count());
    // Full model output, for diagnosing parse failures (non-JSON, markdown fences, reasoning text, …).
    log::debug!("ollama generate raw response: {}", g.response);
    Ok(g.response)
}

/// Read an image file and base64-encode it for the `images` array (standard engine, no data: prefix).
pub fn image_to_base64(path: &str) -> Result<String, String> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    Ok(base64::engine::general_purpose::STANDARD.encode(bytes))
}

/// Candidate `ollama` executables: PATH first, then the platform's default install location — so a
/// missing/stale PATH in the app's environment doesn't hide an installed Ollama or block auto-start.
fn binary_candidates() -> Vec<String> {
    let mut v = vec!["ollama".to_string()];
    #[cfg(target_os = "windows")]
    {
        if let Ok(local) = std::env::var("LOCALAPPDATA") {
            v.push(format!("{local}\\Programs\\Ollama\\ollama.exe"));
        }
    }
    #[cfg(target_os = "macos")]
    {
        v.push("/usr/local/bin/ollama".to_string());
        v.push("/opt/homebrew/bin/ollama".to_string());
        v.push("/Applications/Ollama.app/Contents/Resources/ollama".to_string());
    }
    #[cfg(target_os = "linux")]
    {
        v.push("/usr/local/bin/ollama".to_string());
        v.push("/usr/bin/ollama".to_string());
    }
    v
}

/// Whether the `ollama` command exists on the system (installed), regardless of the daemon running.
/// Blocking — call inside `spawn_blocking`.
pub fn is_installed() -> bool {
    binary_candidates().iter().any(|bin| {
        std::process::Command::new(bin)
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    })
}

/// Ensure the daemon is running before an operation: if the heartbeat fails, spawn `ollama serve`
/// (detached, trying each candidate binary) and poll until it answers (~30 s). Auto-starts Ollama.
pub async fn ensure_running(base_url: &str) -> Result<(), String> {
    if version(base_url).await.is_ok() {
        return Ok(());
    }
    log::info!("ensure_running: Ollama not reachable, starting `ollama serve`");
    let started = binary_candidates().into_iter().any(|bin| {
        std::process::Command::new(&bin)
            .arg("serve")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .is_ok()
    });
    if !started {
        return Err("Ollama is not installed or could not be started".to_string());
    }
    for _ in 0..60 {
        tokio::time::sleep(Duration::from_millis(500)).await;
        if version(base_url).await.is_ok() {
            return Ok(());
        }
    }
    Err("Ollama started but did not become ready in time".to_string())
}
