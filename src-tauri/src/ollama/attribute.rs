//! Attribution: run a vision inference and map the strict JSON onto the editor's metadata.
//! Single mode returns the parsed result (frontend applies it); batch mode stores each result in the
//! metadata store as app-only (feature 008 — the file is not modified), sequentially (Ollama
//! serializes inference), streaming progress.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crate::events::{BatchProgress, ItemStatus};
use crate::store::{DbState, StoredMetadata};

use super::client;
use super::types::{AttributionConfig, AttributionResult};

/// Best-effort extraction of the JSON payload from a model response. Lenient / cloud models often wrap
/// the JSON in a markdown code fence (```json … ```) or surround it with prose; strip the fence and
/// narrow to the outermost object/array span so the parser sees clean JSON.
fn sanitize_keywords(keywords: Vec<String>) -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(25);
    for keyword in keywords {
        let k = keyword.trim();
        if k.is_empty() || !k.is_ascii() {
            continue;
        }
        if out.iter().any(|existing| existing.eq_ignore_ascii_case(k)) {
            continue;
        }
        out.push(k.to_string());
        if out.len() == 25 {
            break;
        }
    }
    out
}

fn normalize_primary_category(result: &mut AttributionResult) {
    if result.categories.first().map(|c| c.as_str()) != Some("Interiors") {
        return;
    }

    let title = result.title.to_ascii_lowercase();
    let first_words = title
        .split_whitespace()
        .take(8)
        .collect::<Vec<_>>()
        .join(" ");

    let living_subject = [
        "cat ", "cats ", "dog ", "dogs ", "woman ", "man ", "person ", "people ",
        "worker ", "doctor ", "nurse ", "caregiver ", "veterinarian ",
    ]
    .iter()
    .any(|prefix| first_words.starts_with(prefix));

    let obvious_pet_product = [
        "collar", "leash", "pet accessory", "pet accessories", "pet bowl", "dog bowl",
        "cat bowl", "pet toy", "cat toy", "dog toy",
    ]
    .iter()
    .any(|term| first_words.contains(term));

    if obvious_pet_product && !living_subject {
        let pet_context = title.contains("pet ")
            || title.contains("dog ")
            || title.contains("cat ")
            || result.keywords.iter().take(10).any(|k| {
                let k = k.to_ascii_lowercase();
                k.contains("pet ") || k.contains("dog ") || k.contains("cat ")
            });

        result.categories = if pet_context {
            vec!["Objects".to_string(), "Animals/Wildlife".to_string()]
        } else {
            vec!["Objects".to_string()]
        };
    }
}

fn keywords_need_retry(keywords: &[String]) -> bool {
    if keywords.len() != 25 {
        return true;
    }
    keywords.iter().any(|k| {
        let trimmed = k.trim();
        trimmed.is_empty() || !trimmed.is_ascii()
    })
}

fn extract_json(raw: &str) -> &str {
    let mut s = raw.trim();

    // Unwrap a fenced code block: ```json … ``` (or a plain ``` … ```).
    if let Some(rest) = s.strip_prefix("```") {
        // Skip the remainder of the opening-fence line (an optional language tag like "json").
        let body = rest.split_once('\n').map_or(rest, |(_, b)| b);
        s = body.trim().strip_suffix("```").unwrap_or(body).trim();
    }

    // Narrow to the outermost JSON object/array span, ignoring any surrounding prose. Brace chars are
    // ASCII, so the byte indices are valid char boundaries.
    let start = s.find(|c| c == '{' || c == '[');
    let end = s.rfind(|c| c == '}' || c == ']');
    match (start, end) {
        (Some(a), Some(b)) if b >= a => &s[a..=b],
        _ => s,
    }
}

/// Parse and validate the model's strict-JSON `response` string into the applied fields. The
/// editorial/mature_content/illustration flags are lenient (default false if missing or non-bool).
fn parse_result(raw: &str) -> Result<AttributionResult, String> {
    let v: serde_json::Value =
        serde_json::from_str(extract_json(raw)).map_err(|e| format!("invalid JSON from model: {e}"))?;
    let str_field = |key: &str| -> Result<String, String> {
        v.get(key)
            .and_then(|x| x.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| format!("response missing string field '{key}'"))
    };
    let arr_field = |key: &str| -> Result<Vec<String>, String> {
        v.get(key)
            .and_then(|x| x.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
            .ok_or_else(|| format!("response missing array field '{key}'"))
    };
    let bool_field = |key: &str| -> bool { v.get(key).and_then(|x| x.as_bool()).unwrap_or(false) };
    let mut result = AttributionResult {
        title: str_field("title")?,
        description: str_field("description")?,
        keywords: sanitize_keywords(arr_field("keywords")?),
        categories: arr_field("categories")?,
        editorial: bool_field("editorial"),
        mature_content: bool_field("mature_content"),
        illustration: bool_field("illustration"),
    };
    normalize_primary_category(&mut result);
    Ok(result)
}

/// Resolve once the shared cancel flag is set (polled). Raced against the inference so a single
/// attribution can be cancelled from the UI — dropping the `generate` future aborts the HTTP request.
async fn cancelled(cancel: &Arc<AtomicBool>) {
    while !cancel.load(Ordering::Relaxed) {
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

/// Single-image attribution → parsed result (the frontend applies it to the form). Auto-starts the
/// daemon and races the inference against the cancel flag so it can be aborted mid-request.
pub async fn attribute_one(
    path: &str,
    cfg: &AttributionConfig,
    cancel: &Arc<AtomicBool>,
) -> Result<AttributionResult, String> {
    log::info!("attribute start: {path} (model {})", cfg.model);
    client::ensure_running(&cfg.base_url).await?;
    let image = client::image_to_base64(path)?;

    // Stage 1: visual evidence only. Do not expose folder or filename context yet.
    // This creates an explicit evidence summary that the final stock-metadata pass must obey.
    let mut facts_cfg = cfg.clone();
    facts_cfg.prompt = r#"Analyze only what is visibly present in this image. Ignore any filename, folder, series, or project context.

Return concise JSON with exactly these fields:
{
  "primary_subject": "short noun phrase",
  "visible_action": "short factual phrase or empty string",
  "visible_objects": ["..."],
  "visible_people_animals": ["..."],
  "visible_setting": "short factual phrase",
  "supported_commercial_scenarios": ["only scenarios directly supported by the image"],
  "unsupported_inferences_to_avoid": ["professions, services, locations, relationships, causes, functions or outcomes that are not visually proven"]
}

Rules:
- Use English only.
- Be conservative and factual.
- Do not infer services, professions, business types, diagnoses, product functions, locations or relationships unless the image clearly supports them.
- Do not use marketing language.
- Keep the JSON compact.
"#.to_string();
    facts_cfg.format = serde_json::json!({
        "type": "object",
        "properties": {
            "primary_subject": {"type":"string"},
            "visible_action": {"type":"string"},
            "visible_objects": {"type":"array","items":{"type":"string"}},
            "visible_people_animals": {"type":"array","items":{"type":"string"}},
            "visible_setting": {"type":"string"},
            "supported_commercial_scenarios": {"type":"array","items":{"type":"string"}},
            "unsupported_inferences_to_avoid": {"type":"array","items":{"type":"string"}}
        },
        "required": [
            "primary_subject","visible_action","visible_objects","visible_people_animals",
            "visible_setting","supported_commercial_scenarios","unsupported_inferences_to_avoid"
        ]
    });

    let visual_facts_raw = tokio::select! {
        _ = cancelled(cancel) => return Err("cancelled".to_string()),
        result = client::generate(&facts_cfg, image.clone(), path, false) => result?,
    };
    let visual_facts = extract_json(&visual_facts_raw).to_string();

    // Stage 2: validate filename/folder context against the visual evidence.
    // Filename is strong item-specific context; folder is softer series/global context.
    let (folder_context, filename_context) = client::series_context(path);
    let mut context_cfg = cfg.clone();
    context_cfg.prompt = format!(
        r#"You are validating contextual metadata against an image evidence summary.

VISUAL EVIDENCE:
{}

FILENAME CONTEXT (strong, item-specific):
{}

FOLDER CONTEXT (soft, series/global):
{}

Return compact JSON with exactly these fields:
{{
  "trusted_filename_facts": ["explicit filename roles, objects, or actions that are not contradicted by the image"],
  "supported_folder_concepts": ["folder concepts that are genuinely compatible with the visual evidence"],
  "rejected_context_concepts": ["folder/filename concepts that would be unsupported, over-specific, or contradictory"],
  "context_support": "high|partial|none"
}}

Rules:
- Explicit filename roles/actions are trusted unless the image clearly contradicts them.
- Do not upgrade filename roles. "chef" does not mean executive chef; "server" does not mean sous chef.
- Folder context is secondary by default, BUT if filename and folder independently describe the same commercial scenario and the image is compatible, set context_support to "high".
- HIGH CONFIDENCE CONTEXT means the shared filename+folder scenario should become the primary commercial interpretation for title, description, and top keywords.
- Example: filename "Manager_briefing_waiters_at_table" + folder meaning staff meetings / pre-shift instruction + image showing one central person with multiple restaurant staff taking notes => validate manager, waiters, staff briefing, pre-shift briefing, restaurant staff training/operations as the intended scenario.
- When context_support is "high", reject contradictory alternative roles or relationships that are not supported by filename+folder+image. Example: do not call the central person a customer when filename says manager and surrounding people are waiters/staff.
- If filename and folder do NOT reinforce each other, keep folder context soft and validate only the parts supported by the image.
- Reject inferred professions, services, place types, relationships, diagnoses, product functions, causes, or outcomes that are not supported.
- If an image only shows an object such as a ball, do not validate dog daycare, pet boarding, pet walking area, or staff/service concepts unless visible evidence supports them.
- Use English only.
"#,
        visual_facts, filename_context, folder_context
    );
    context_cfg.format = serde_json::json!({
        "type": "object",
        "properties": {
            "trusted_filename_facts": {"type":"array","items":{"type":"string"}},
            "supported_folder_concepts": {"type":"array","items":{"type":"string"}},
            "rejected_context_concepts": {"type":"array","items":{"type":"string"}},
            "context_support": {"type":"string","enum":["high","partial","none"]}
        },
        "required": [
            "trusted_filename_facts","supported_folder_concepts",
            "rejected_context_concepts","context_support"
        ]
    });

    let validated_context_raw = tokio::select! {
        _ = cancelled(cancel) => return Err("cancelled".to_string()),
        result = client::generate(&context_cfg, image.clone(), path, false) => result?,
    };
    let validated_context = extract_json(&validated_context_raw).to_string();

    // Stage 3: final metadata. Feed only visual facts + validated context.
    // Raw folder/filename are deliberately NOT appended to this request.

    let mut final_cfg = cfg.clone();
    final_cfg.prompt.push_str(
        "\n\nVISUAL EVIDENCE PASS (AUTHORITATIVE):\n"
    );
    final_cfg.prompt.push_str(&visual_facts);
    final_cfg.prompt.push_str(
        "\n\nVALIDATED CONTEXT:\n"
    );
    final_cfg.prompt.push_str(&validated_context);
    final_cfg.prompt.push_str(
        "\n\nFINAL ENRICHMENT RULES:\nUse the VISUAL EVIDENCE PASS as the source of truth for what is visibly present. Use trusted_filename_facts actively because they describe this exact item unless the image contradicts them. Never use rejected_context_concepts. Do not see or infer any raw folder/filename beyond this validated context. If context_support is high, the combination of trusted_filename_facts + supported_folder_concepts is the PRIMARY COMMERCIAL SCENARIO and should drive the title, description, and first 10 keywords. In high-confidence cases, prefer the validated scenario over a generic literal interpretation of the image. Also suppress contradictory alternatives: if validated context says manager briefing waiters, do not describe the manager as a customer, guest, diner, or unrelated staff member. If context_support is partial, use supported_folder_concepts only as secondary enrichment. If context_support is none, ignore folder concepts entirely. The first 10 keywords should prioritize the validated scenario when high-confidence; otherwise prioritize trusted filename facts plus the visible primary subject, action, and objects. Do not upgrade roles or relationships beyond the validated context. Category must follow the primary commercial subject after combining visible facts with trusted filename facts and, when high-confidence, supported folder concepts."
    );

    let raw = tokio::select! {
        _ = cancelled(cancel) => return Err("cancelled".to_string()),
        result = client::generate(&final_cfg, image.clone(), path, false) => result?,
    };
    let mut result = parse_result(&raw)?;

    // Some multilingual vision models occasionally ignore an English-only prompt and emit CJK
    // keywords. Enforce the stock workflow at the application boundary instead of trusting the model.
    // If sanitizing leaves anything other than exactly 25 ASCII keywords, retry once with a strict
    // corrective instruction, then sanitize again.
    if keywords_need_retry(&result.keywords) {
        log::warn!(
            "attribute keyword validation failed for {path}: {} valid English keywords; retrying once",
            result.keywords.len()
        );
        let mut retry_cfg = final_cfg.clone();
        retry_cfg.prompt.push_str(
            "\n\nCRITICAL RETRY RULE: Return exactly 25 keywords. Every keyword must be English and ASCII-only. Do not use Chinese, Cyrillic, accented characters, mixed-language text, or malformed tokens. Use natural stock-buyer search phrases only."
        );
        let retry_raw = tokio::select! {
            _ = cancelled(cancel) => return Err("cancelled".to_string()),
            retry = client::generate(&retry_cfg, image, path, false) => retry?,
        };
        result = parse_result(&retry_raw)?;
    }

    if result.keywords.len() != 25 {
        return Err(format!(
            "model returned only {} valid English keywords after retry; expected exactly 25",
            result.keywords.len()
        ));
    }

    log::info!(
        "attribute done: {path} → {} keywords, {} categories",
        result.keywords.len(),
        result.categories.len()
    );
    Ok(result)
}

/// Attribute one file and store the result in the metadata store as app-only (feature 008 — the file
/// is NOT modified). Keyword merge + release_filename preservation happen inside `store_attribution`.
/// Returns the (unchanged) path.
async fn attribute_and_store(
    path: &str,
    cfg: &AttributionConfig,
    cancel: &Arc<AtomicBool>,
    db: &DbState,
) -> Result<String, String> {
    let result = attribute_one(path, cfg, cancel).await?;
    let model = StoredMetadata {
        title: result.title,
        description: result.description,
        keywords: result.keywords,
        categories: result.categories.join(", "),
        release_filename: String::new(),
        editorial: result.editorial,
        mature_content: result.mature_content,
        illustration: result.illustration,
    };
    // Run the SQLite write off the async runtime (no mutex/SQLite work on a tokio worker thread).
    let dbh = db.share();
    let p = path.to_string();
    tokio::task::spawn_blocking(move || dbh.store_attribution(&p, &model))
        .await
        .map_err(|e| e.to_string())?;
    Ok(path.to_string())
}

/// Sequentially attribute every path and store each result (app-only), streaming one `BatchProgress`
/// per file. A failed item is recorded and the loop continues; cancellation stops before the next item.
pub async fn attribute_batch(
    paths: &[String],
    cfg: &AttributionConfig,
    cancel: &Arc<AtomicBool>,
    db: &DbState,
    progress: impl Fn(BatchProgress),
) -> Vec<ItemStatus> {
    let mut out = Vec::with_capacity(paths.len());
    for (index, path) in paths.iter().enumerate() {
        let status = if cancel.load(Ordering::Relaxed) {
            ItemStatus::Cancelled
        } else {
            match attribute_and_store(path, cfg, cancel, db).await {
                Ok(p) => ItemStatus::Ok { path: p },
                // Cancelled mid-inference (the generate future was dropped) — record it as cancelled.
                Err(_) if cancel.load(Ordering::Relaxed) => ItemStatus::Cancelled,
                Err(error) => {
                    log::warn!("attribute batch item {index} failed: {error}");
                    ItemStatus::Failed { error }
                }
            }
        };
        progress(BatchProgress { index, status: status.clone() });
        out.push(status);
    }
    out
}
