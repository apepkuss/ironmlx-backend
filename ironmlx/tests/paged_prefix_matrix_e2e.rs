//! Real-checkpoint matrix E2E for paged SSD prefix cache.
//!
//! These tests intentionally run the public `ironmlx serve` CLI and are ignored
//! by default because they require local checkpoint snapshots.
//!
//! ```text
//! MLX_DIR=$HOME/.local/mlx \
//! QWEN35_MODEL=/path/to/Qwen3.5-4B-MLX-4bit/snapshots/<sha> \
//! QWEN36_DENSE_MODEL=/path/to/Qwen3.6-27B-4bit/snapshots/<sha> \
//! QWEN36_DENSE_MTP_MODEL=/path/to/Qwen3.6-27B-MTP-4bit/snapshots/<sha> \
//! QWEN38_DENSE_MODEL=/path/to/Qwen3.8-27B-4bit/snapshots/<sha> \
//! QWEN38_DENSE_MTP_MODEL=/path/to/Qwen3.8-27B-MTP-4bit/snapshots/<sha> \
//! GLM47_MODEL_DIR=/path/to/GLM-4.7-Flash-4bit/snapshots/<sha> \
//! cargo test --release -p ironmlx --test paged_prefix_matrix_e2e \
//!   -- --ignored --test-threads=1 --nocapture
//! ```

use std::collections::BTreeSet;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use base64::Engine;
use ironmlx::core::scheduler::DenseVlMethods;
use ironmlx::core::server::chat_format::render_and_encode;
use ironmlx::core::server::vision::{expand_decoded_messages, DecodedMessage, DecodedPart};
use ironmlx::core::server::VisionInputConfig;
use ironmlx::core::{Loader, Model, Tokenizer};
use ironmlx::nn::enable_paged_kv_caches;
use mlx::Dtype;

struct ServerProcess {
    child: Child,
    stderr: Arc<Mutex<Vec<u8>>>,
    drainer: Option<JoinHandle<()>>,
}

impl ServerProcess {
    fn spawn(model_dir: &Path, cache_dir: &Path, port: u16) -> Self {
        Self::spawn_with_max_sequences(model_dir, cache_dir, port, 2)
    }

    fn spawn_with_max_sequences(
        model_dir: &Path,
        cache_dir: &Path,
        port: u16,
        max_sequences: usize,
    ) -> Self {
        Self::spawn_with_options(model_dir, cache_dir, None, None, None, port, max_sequences)
    }

    fn spawn_with_kv_quant(
        model_dir: &Path,
        cache_dir: &Path,
        kv_quant: &str,
        port: u16,
        max_sequences: usize,
    ) -> Self {
        Self::spawn_with_options(
            model_dir,
            cache_dir,
            Some(kv_quant),
            None,
            None,
            port,
            max_sequences,
        )
    }

    fn spawn_with_active_kv(
        model_dir: &Path,
        cache_dir: &Path,
        active_kv_dir: &Path,
        port: u16,
        max_sequences: usize,
    ) -> Self {
        Self::spawn_with_options(
            model_dir,
            cache_dir,
            None,
            Some(active_kv_dir),
            None,
            port,
            max_sequences,
        )
    }

    fn spawn_with_mtp_and_active_kv(
        model_dir: &Path,
        mtp_model_dir: &Path,
        cache_dir: &Path,
        active_kv_dir: &Path,
        port: u16,
        max_sequences: usize,
    ) -> Self {
        Self::spawn_with_options(
            model_dir,
            cache_dir,
            None,
            Some(active_kv_dir),
            Some(mtp_model_dir),
            port,
            max_sequences,
        )
    }

    fn spawn_with_mtp_long_context(
        model_dir: &Path,
        mtp_model_dir: &Path,
        cache_dir: &Path,
        port: u16,
        max_cache_cap: usize,
    ) -> Self {
        Self::spawn_with_options_and_max_cache_cap(
            model_dir,
            cache_dir,
            None,
            None,
            Some(mtp_model_dir),
            port,
            1,
            max_cache_cap,
            2,
        )
    }

    fn spawn_with_mtp_draft_tokens(
        model_dir: &Path,
        mtp_model_dir: &Path,
        cache_dir: &Path,
        port: u16,
        draft_tokens: usize,
    ) -> Self {
        Self::spawn_with_options_and_max_cache_cap(
            model_dir,
            cache_dir,
            None,
            None,
            Some(mtp_model_dir),
            port,
            1,
            4_096,
            draft_tokens,
        )
    }

    fn spawn_with_kv_quant_and_active_kv(
        model_dir: &Path,
        cache_dir: &Path,
        kv_quant: &str,
        active_kv_dir: &Path,
        port: u16,
        max_sequences: usize,
    ) -> Self {
        Self::spawn_with_options(
            model_dir,
            cache_dir,
            Some(kv_quant),
            Some(active_kv_dir),
            None,
            port,
            max_sequences,
        )
    }

    fn spawn_with_options(
        model_dir: &Path,
        cache_dir: &Path,
        kv_quant: Option<&str>,
        active_kv_dir: Option<&Path>,
        mtp_model_dir: Option<&Path>,
        port: u16,
        max_sequences: usize,
    ) -> Self {
        Self::spawn_with_options_and_max_cache_cap(
            model_dir,
            cache_dir,
            kv_quant,
            active_kv_dir,
            mtp_model_dir,
            port,
            max_sequences,
            4_096,
            2,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn spawn_with_options_and_max_cache_cap(
        model_dir: &Path,
        cache_dir: &Path,
        kv_quant: Option<&str>,
        active_kv_dir: Option<&Path>,
        mtp_model_dir: Option<&Path>,
        port: u16,
        max_sequences: usize,
        max_cache_cap: usize,
        mtp_draft_tokens: usize,
    ) -> Self {
        let bin = env!("CARGO_BIN_EXE_ironmlx");
        let mlx_dir = std::env::var("MLX_DIR").expect("MLX_DIR must be set");
        let prefix_cache_max_pages = max_cache_cap.div_ceil(16).max(4_096);
        let mut args = vec![
            "serve".to_owned(),
            "--model".to_owned(),
            model_dir
                .to_str()
                .expect("model path must be valid UTF-8")
                .to_owned(),
            "--paged-prefix-cache-dir".to_owned(),
            cache_dir
                .to_str()
                .expect("prefix cache path must be valid UTF-8")
                .to_owned(),
            "--paged-prefix-cache-block-size".to_owned(),
            "16".to_owned(),
            "--paged-prefix-cache-max-pages".to_owned(),
            prefix_cache_max_pages.to_string(),
            "--host".to_owned(),
            "127.0.0.1".to_owned(),
            "--port".to_owned(),
            port.to_string(),
            "--max-sequences".to_owned(),
            max_sequences.to_string(),
            "--admission-deadline-ms".to_owned(),
            "200".to_owned(),
            "--admission-queue-max".to_owned(),
            "8".to_owned(),
            "--prefill-chunk-size".to_owned(),
            "0".to_owned(),
            "--max-cache-cap".to_owned(),
            max_cache_cap.to_string(),
        ];
        if let Some(kv_quant) = kv_quant {
            args.push("--kv-quant".to_owned());
            args.push(kv_quant.to_owned());
        }
        if let Some(active_kv_dir) = active_kv_dir {
            args.push("--active-kv-offload".to_owned());
            args.push("--active-kv-offload-dir".to_owned());
            args.push(
                active_kv_dir
                    .to_str()
                    .expect("active KV path must be valid UTF-8")
                    .to_owned(),
            );
        }
        if let Some(mtp_model_dir) = mtp_model_dir {
            args.push("--mtp-model-dir".to_owned());
            args.push(
                mtp_model_dir
                    .to_str()
                    .expect("MTP model path must be valid UTF-8")
                    .to_owned(),
            );
            args.push("--mtp-draft-tokens".to_owned());
            args.push(mtp_draft_tokens.to_string());
        }
        let mut cmd = Command::new(bin);
        cmd.current_dir(env!("CARGO_MANIFEST_DIR"));
        cmd.args(args);
        cmd.env("MLX_DIR", mlx_dir);
        cmd.env("RUST_LOG", "ironmlx=debug,warn");
        cmd.stderr(Stdio::piped());
        cmd.stdout(Stdio::null());
        let mut child = cmd.spawn().expect("spawn ironmlx serve");
        let stderr = Arc::new(Mutex::new(Vec::<u8>::new()));
        let stderr_pipe = child.stderr.take().expect("server stderr");
        let stderr_for_thread = Arc::clone(&stderr);
        let drainer = std::thread::spawn(move || {
            let mut reader = BufReader::new(stderr_pipe);
            let mut line = String::new();
            loop {
                line.clear();
                match reader.read_line(&mut line) {
                    Ok(0) => break,
                    Ok(_) => stderr_for_thread
                        .lock()
                        .expect("stderr lock")
                        .extend_from_slice(line.as_bytes()),
                    Err(_) => break,
                }
            }
        });
        Self {
            child,
            stderr,
            drainer: Some(drainer),
        }
    }

    fn stderr_text(&self) -> String {
        String::from_utf8_lossy(&self.stderr.lock().expect("stderr lock")).to_string()
    }
}

impl Drop for ServerProcess {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(drainer) = self.drainer.take() {
            let _ = drainer.join();
        }
    }
}

fn unique_temp_dir(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("ironmlx-{name}-{}-{nanos}", std::process::id()))
}

async fn alloc_port() -> u16 {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral port");
    let port = listener.local_addr().expect("local addr").port();
    drop(listener);
    port
}

fn snapshot_from_env_or_default(env_name: &str, repo_dir: &str) -> PathBuf {
    if let Ok(path) = std::env::var(env_name) {
        let path = PathBuf::from(path);
        assert!(
            path.exists(),
            "{env_name} does not exist: {}",
            path.display()
        );
        return path;
    }
    let home = std::env::var("HOME").expect("HOME env");
    let snapshots = PathBuf::from(home)
        .join(".ironmlx/models")
        .join(repo_dir)
        .join("snapshots");
    let first = std::fs::read_dir(&snapshots)
        .unwrap_or_else(|err| panic!("read {}: {err}", snapshots.display()))
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .find(|path| path.is_dir())
        .unwrap_or_else(|| panic!("no snapshot dirs under {}", snapshots.display()));
    assert!(first.join("config.json").exists(), "missing config.json");
    first
}

fn qwen35_model_dir() -> PathBuf {
    snapshot_from_env_or_default("QWEN35_MODEL", "models--mlx-community--Qwen3.5-4B-MLX-4bit")
}

fn qwen36_dense_model_dir() -> PathBuf {
    snapshot_from_env_or_default(
        "QWEN36_DENSE_MODEL",
        "huggingface/mlx-community--Qwen3.6-27B-4bit",
    )
}

fn qwen36_dense_mtp_model_dir() -> PathBuf {
    snapshot_from_env_or_default(
        "QWEN36_DENSE_MTP_MODEL",
        "huggingface/mlx-community--Qwen3.6-27B-MTP-4bit",
    )
}

fn qwen38_dense_model_dir() -> PathBuf {
    snapshot_from_env_or_default(
        "QWEN38_DENSE_MODEL",
        "huggingface/mlx-community--Qwen3.8-27B-4bit",
    )
}

fn qwen38_dense_mtp_model_dir() -> PathBuf {
    snapshot_from_env_or_default(
        "QWEN38_DENSE_MTP_MODEL",
        "huggingface/mlx-community--Qwen3.8-27B-MTP-4bit",
    )
}

fn gemma4_unified_model_dir() -> PathBuf {
    snapshot_from_env_or_default(
        "GEMMA4_LONG_CONTEXT_MODEL",
        "huggingface/mlx-community--gemma-4-12B-it-4bit",
    )
}

fn gemma4_unified_drafter_model_dir() -> PathBuf {
    snapshot_from_env_or_default(
        "GEMMA4_LONG_CONTEXT_DRAFTER",
        "huggingface/mlx-community--gemma-4-12B-it-assistant-4bit",
    )
}

fn glm47_model_dir() -> PathBuf {
    snapshot_from_env_or_default(
        "GLM47_MODEL_DIR",
        "models--mlx-community--GLM-4.7-Flash-4bit",
    )
}

fn minicpmv46_model_dir() -> PathBuf {
    snapshot_from_env_or_default(
        "MINICPMV46_MODEL",
        "models--mlx-community--MiniCPM-V-4.6-4bit",
    )
}

fn gemma4_model_dir() -> PathBuf {
    snapshot_from_env_or_default("GEMMA4_MODEL", "models--mlx-community--gemma-4-e4b-it-4bit")
}

fn gemma4_moe_model_dir() -> PathBuf {
    snapshot_from_env_or_default(
        "GEMMA4_MOE_MODEL",
        "models--mlx-community--gemma-4-26b-a4b-it-4bit",
    )
}

fn coco_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("qwen35_vl")
        .join("coco_sample.jpg")
}

fn image_data_url(path: &Path) -> String {
    let bytes = std::fs::read(path).expect("read image fixture");
    format!(
        "data:image/jpeg;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(bytes)
    )
}

fn write_flipped_image(src: &Path, dst: &Path) {
    let img = image::open(src).expect("open source image");
    let flipped =
        image::DynamicImage::ImageRgba8(image::imageops::flip_horizontal(&img.to_rgba8()));
    flipped
        .save_with_format(dst, image::ImageFormat::Jpeg)
        .expect("write flipped image");
}

fn text_body(prompt: &str) -> serde_json::Value {
    serde_json::json!({
        "model": "paged-prefix-matrix",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 3,
        "temperature": 0.0,
        "stream": false
    })
}

fn vl_body(image_path: &Path) -> serde_json::Value {
    serde_json::json!({
        "model": "paged-prefix-matrix",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the image."},
                {"type": "image_url", "image_url": {"url": image_data_url(image_path)}}
            ]
        }],
        "max_tokens": 16,
        "temperature": 0.0,
        "stream": false
    })
}

fn client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(300))
        .no_proxy()
        .build()
        .expect("reqwest client")
}

fn long_context_client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(1_200))
        .no_proxy()
        .build()
        .expect("long-context reqwest client")
}

fn long_context_prompt(model_dir: &Path, minimum_tokens: usize) -> String {
    let tokenizer = Tokenizer::from_model_dir(model_dir).expect("load long-context tokenizer");
    let unit = " exact verification";
    let unit_tokens = tokenizer
        .encode(unit, false)
        .expect("encode long-context unit")
        .len()
        .max(1);
    let mut prompt = unit.repeat(minimum_tokens.div_ceil(unit_tokens));
    while tokenizer
        .encode(&prompt, false)
        .expect("encode long-context prompt")
        .len()
        < minimum_tokens
    {
        prompt.push_str(unit);
    }
    prompt.push_str("\nAnswer with a short deterministic sentence.");
    prompt
}

async fn wait_ready(client: &reqwest::Client, port: u16, server: &mut ServerProcess) {
    let deadline = Instant::now() + Duration::from_secs(240);
    loop {
        match client
            .get(format!("http://127.0.0.1:{port}/health"))
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() => return,
            _ => {}
        }
        if let Some(status) = server.child.try_wait().expect("server try_wait") {
            panic!(
                "ironmlx serve exited before ready: {status}; stderr:\n{}",
                server.stderr_text()
            );
        }
        assert!(
            Instant::now() < deadline,
            "server did not become ready; stderr:\n{}",
            server.stderr_text()
        );
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}

async fn post_chat(
    client: &reqwest::Client,
    port: u16,
    body: serde_json::Value,
) -> serde_json::Value {
    let resp = client
        .post(format!("http://127.0.0.1:{port}/v1/chat/completions"))
        .json(&body)
        .send()
        .await
        .expect("chat completion send");
    let status = resp.status();
    let text = resp.text().await.expect("chat completion body");
    assert_eq!(status, 200, "chat completion failed: {text}");
    let json: serde_json::Value = serde_json::from_str(&text).expect("chat completion json");
    assert!(
        json["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or_default()
            .len()
            > 0,
        "assistant content must not be empty: {json}"
    );
    assert!(
        json["usage"]["prompt_tokens"].as_u64().unwrap_or_default() > 0,
        "prompt token count must be present: {json}"
    );
    json
}

async fn post_chat_with_server_diagnostics(
    client: &reqwest::Client,
    port: u16,
    body: serde_json::Value,
    server: &ServerProcess,
) -> serde_json::Value {
    let resp = client
        .post(format!("http://127.0.0.1:{port}/v1/chat/completions"))
        .json(&body)
        .send()
        .await
        .expect("chat completion send");
    let status = resp.status();
    let text = resp.text().await.expect("chat completion body");
    assert_eq!(
        status,
        200,
        "chat completion failed: {text}; server stderr:\n{}",
        server.stderr_text()
    );
    let json: serde_json::Value = serde_json::from_str(&text).expect("chat completion json");
    assert!(
        json["choices"][0]["message"]["content"]
            .as_str()
            .is_some_and(|content| !content.is_empty()),
        "assistant content must not be empty: {json}"
    );
    assert!(
        json["usage"]["prompt_tokens"].as_u64().unwrap_or_default() > 0,
        "prompt token count must be present: {json}"
    );
    json
}

async fn healthz(client: &reqwest::Client, port: u16) -> serde_json::Value {
    let resp = client
        .get(format!("http://127.0.0.1:{port}/healthz"))
        .send()
        .await
        .expect("healthz send");
    let status = resp.status();
    let text = resp.text().await.expect("healthz body");
    assert_eq!(status, 200, "healthz failed: {text}");
    serde_json::from_str(&text).expect("healthz json")
}

fn assert_active_kv_health(health: &serde_json::Value) {
    assert_eq!(
        health["memory"]["kv_cache_budget_policy"].as_str(),
        Some("active_kv_offload"),
        "unexpected healthz memory policy: {health}"
    );
    assert_eq!(
        health["memory"]["kv_cache_logical_cap_tokens"].as_u64(),
        Some(4096),
        "unexpected logical cap: {health}"
    );
    assert_eq!(
        health["memory"]["kv_cache_resident_cap_tokens"].as_u64(),
        Some(1024),
        "unexpected resident cap: {health}"
    );
    assert_eq!(
        health["active_kv_offload"]["enabled"].as_bool(),
        Some(true),
        "Active KV should be enabled: {health}"
    );
    assert_eq!(
        health["active_kv_offload"]["degraded"].as_bool(),
        Some(false),
        "Active KV should not be degraded: {health}"
    );
    assert_eq!(
        health["active_kv_offload"]["swap_error_count"].as_u64(),
        Some(0),
        "Active KV swap errors: {health}"
    );
}

fn assert_mtp_draft_width(health: &serde_json::Value, expected: u64) {
    assert_eq!(
        health["mtp"]["enabled"].as_bool(),
        Some(true),
        "MTP must be enabled: {health}"
    );
    assert_eq!(
        health["mtp"]["requested_draft_tokens"].as_u64(),
        Some(expected),
        "unexpected requested MTP draft width: {health}"
    );
    assert_eq!(
        health["mtp"]["draft_tokens"].as_u64(),
        Some(expected),
        "runtime must preserve the requested MTP draft width: {health}"
    );
}

async fn post_same_body_concurrently(
    client: &reqwest::Client,
    port: u16,
    body: serde_json::Value,
    concurrency: usize,
) -> Vec<serde_json::Value> {
    let mut tasks = Vec::with_capacity(concurrency);
    for _ in 0..concurrency {
        let client = client.clone();
        let body = body.clone();
        tasks.push(tokio::spawn(
            async move { post_chat(&client, port, body).await },
        ));
    }

    let mut responses = Vec::with_capacity(concurrency);
    for task in tasks {
        responses.push(task.await.expect("concurrent chat task"));
    }
    responses
}

fn count_log(text: &str, needle: &str) -> usize {
    text.match_indices(needle).count()
}

async fn wait_log_count(server: &ServerProcess, needle: &str, min_count: usize, timeout: Duration) {
    let deadline = Instant::now() + timeout;
    loop {
        let text = server.stderr_text();
        if count_log(&text, needle) >= min_count {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "log needle {needle:?} did not reach {min_count}; stderr:\n{text}"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

fn cache_entry_dirs(cache_dir: &Path) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    for entry in std::fs::read_dir(cache_dir).expect("read prefix cache dir") {
        let path = entry.expect("cache dir entry").path();
        if path.is_dir() {
            dirs.push(path);
        }
    }
    dirs
}

fn cache_metadata(cache_dir: &Path) -> Vec<serde_json::Value> {
    cache_entry_dirs(cache_dir)
        .into_iter()
        .map(|entry| {
            let meta_path = entry.join("meta.json");
            serde_json::from_slice(&std::fs::read(&meta_path).expect("read cache meta"))
                .unwrap_or_else(|err| panic!("cache meta json {}: {err}", meta_path.display()))
        })
        .collect()
}

fn cache_main_layer_kinds(cache_dir: &Path) -> BTreeSet<String> {
    let mut kinds = BTreeSet::new();
    for meta in cache_metadata(cache_dir) {
        let layers = meta["main_layers"]
            .as_array()
            .expect("main_layers must be an array");
        assert!(!layers.is_empty(), "main_layers must not be empty: {meta}");
        for layer in layers {
            kinds.insert(layer["kind"].as_str().expect("main layer kind").to_owned());
        }
    }
    kinds
}

fn assert_cache_kinds(cache_dir: &Path, expected: &[&str]) {
    let kinds = cache_main_layer_kinds(cache_dir);
    for kind in expected {
        assert!(
            kinds.contains(*kind),
            "expected prefix cache kind {kind:?}; actual kinds={kinds:?}"
        );
    }
}

fn assert_has_vl_fingerprint(cache_dir: &Path) {
    let metas = cache_metadata(cache_dir);
    assert!(
        metas.iter().any(|meta| !meta["fingerprint_hash"].is_null()),
        "expected at least one VL prefix cache entry with fingerprint_hash: {metas:?}"
    );
}

fn greedy_argmax(logits: &mlx::Array) -> u32 {
    let f32_logits = logits.astype(Dtype::Float32).expect("astype f32");
    let values = f32_logits.to_vec::<f32>().expect("logits to_vec");
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, _)| idx as u32)
        .expect("non-empty logits")
}

fn slice_vision_rows(embeds: &mlx::Array, start: i32, end: i32) -> mlx::Array {
    let shape = embeds.shape();
    let dims = shape.as_slice();
    assert_eq!(dims.len(), 2, "vision embeds must be [N,H], got {dims:?}");
    mlx::ops::indexing::slice_strided(
        embeds,
        &[start, 0_i32][..],
        &[end, dims[1]][..],
        &[1_i32, 1][..],
    )
    .expect("slice vision rows")
}

fn gemma4_prompt_inputs(
    tokenizer: &Tokenizer,
    vision: &VisionInputConfig,
) -> (Vec<i32>, Vec<mlx::Array>, Vec<(i32, i32, i32)>, i32) {
    let image_token_id =
        ironmlx::core::server::vision::derive_image_token_and_merge(vision, tokenizer).0;
    let bytes = std::fs::read(coco_path()).expect("read image fixture");
    let messages = vec![DecodedMessage {
        role: "user".to_owned(),
        parts: vec![
            DecodedPart::Text("Describe the image.".to_owned()),
            DecodedPart::Image(bytes),
        ],
        reasoning_content: None,
    }];
    let (flat_messages, pixel_values, grid_thw) =
        expand_decoded_messages(messages, vision).expect("expand Gemma4 VL messages");
    let prompt_ids = render_and_encode(tokenizer, &flat_messages, None)
        .expect("render/tokenize Gemma4 prompt")
        .into_iter()
        .map(|id| id as i32)
        .collect::<Vec<_>>();
    (
        prompt_ids,
        pixel_values.expect("Gemma4 prompt has image"),
        grid_thw,
        image_token_id,
    )
}

fn gemma4_split_prefill_argmax(
    model: &ironmlx::models::Gemma4Model,
    prompt_ids: &[i32],
    pixel_values: &[mlx::Array],
    grid_thw: &[(i32, i32, i32)],
    image_token_id: i32,
    paged: bool,
) -> u32 {
    let cap = i32::try_from(prompt_ids.len() + 16).expect("prompt len fits i32");
    let mut cache = model
        .make_cache(1, cap, model.cache_dtype())
        .expect("make Gemma4 cache");
    if paged {
        enable_paged_kv_caches(&mut cache, 16, 4096).expect("enable paged KV cache");
    }

    let vision_embeds = model
        .compute_vision_embeds(pixel_values, grid_thw, mlx::StreamOrDevice::default())
        .expect("compute Gemma4 vision embeds");
    let prefix_len = prompt_ids.len() - 1;
    let prefix_ids = &prompt_ids[..prefix_len];
    let last_ids = &prompt_ids[prefix_len..];
    let prefix_image_pads = prefix_ids
        .iter()
        .filter(|&&tok| tok == image_token_id)
        .count() as i32;
    let last_image_pads = last_ids
        .iter()
        .filter(|&&tok| tok == image_token_id)
        .count() as i32;
    let prefix_vision =
        (prefix_image_pads > 0).then(|| slice_vision_rows(&vision_embeds, 0, prefix_image_pads));
    let last_vision = (last_image_pads > 0).then(|| {
        slice_vision_rows(
            &vision_embeds,
            prefix_image_pads,
            prefix_image_pads + last_image_pads,
        )
    });

    let prefix_input: mlx::Array = (prefix_ids, &[1_i32, prefix_len as i32][..])
        .try_into()
        .expect("prefix input");
    let prefix_pos_data = vec![0_i32; prefix_len];
    let prefix_pos: mlx::Array = (prefix_pos_data.as_slice(), &[1_i32, prefix_len as i32][..])
        .try_into()
        .expect("prefix dummy pos");
    let prefix_hidden = model
        .forward_vl_hidden(
            &prefix_input,
            &prefix_pos,
            None,
            None,
            Some(&mut cache),
            prefix_vision.as_ref(),
            image_token_id,
            mlx::StreamOrDevice::default(),
        )
        .expect("Gemma4 prefix hidden");
    mlx::transforms::eval(&[&prefix_hidden]).expect("eval prefix hidden");

    let last_input: mlx::Array = (last_ids, &[1_i32, 1_i32][..])
        .try_into()
        .expect("last input");
    let last_pos: mlx::Array = (&[0_i32][..], &[1_i32, 1_i32][..])
        .try_into()
        .expect("last dummy pos");
    let logits = model
        .forward_vl_chunk(
            &last_input,
            &last_pos,
            None,
            None,
            Some(&mut cache),
            last_vision.as_ref(),
            image_token_id,
            mlx::StreamOrDevice::default(),
        )
        .expect("Gemma4 last logits");
    mlx::transforms::eval(&[&logits]).expect("eval last logits");
    greedy_argmax(&logits)
}

async fn run_text_exact_hit_case(model_dir: PathBuf, expected_kinds: &[&str], prompt: &str) {
    let cache_dir = unique_temp_dir("text-prefix-matrix");
    std::fs::create_dir_all(&cache_dir).expect("create prefix cache dir");
    let port = alloc_port().await;
    let mut server = ServerProcess::spawn(&model_dir, &cache_dir, port);
    let client = client();
    wait_ready(&client, port, &mut server).await;

    let body = text_body(prompt);
    let warm = post_chat(&client, port, body.clone()).await;
    assert_cache_kinds(&cache_dir, expected_kinds);

    let (hit_a, hit_b) = tokio::join!(
        post_chat(&client, port, body.clone()),
        post_chat(&client, port, body.clone())
    );
    assert_eq!(
        hit_a["usage"]["prompt_tokens"],
        warm["usage"]["prompt_tokens"]
    );
    assert_eq!(
        hit_b["usage"]["prompt_tokens"],
        warm["usage"]["prompt_tokens"]
    );
    wait_log_count(
        &server,
        "paged SSD prefix cache hit",
        2,
        Duration::from_secs(10),
    )
    .await;
    assert_cache_kinds(&cache_dir, expected_kinds);

    std::fs::remove_dir_all(&cache_dir).expect("cleanup prefix cache dir");
}

async fn run_text_concurrent_exact_hit_case(
    model_dir: PathBuf,
    expected_kinds: &[&str],
    prompt: &str,
    concurrency: usize,
) {
    let cache_dir = unique_temp_dir("text-prefix-concurrent");
    std::fs::create_dir_all(&cache_dir).expect("create prefix cache dir");
    let port = alloc_port().await;
    let mut server =
        ServerProcess::spawn_with_max_sequences(&model_dir, &cache_dir, port, concurrency);
    let client = client();
    wait_ready(&client, port, &mut server).await;

    let body = text_body(prompt);
    let warm = post_chat(&client, port, body.clone()).await;
    assert_cache_kinds(&cache_dir, expected_kinds);

    let hits_before = count_log(&server.stderr_text(), "paged SSD prefix cache hit");
    let responses = post_same_body_concurrently(&client, port, body, concurrency).await;
    for response in responses {
        assert_eq!(
            response["usage"]["prompt_tokens"], warm["usage"]["prompt_tokens"],
            "concurrent exact-hit prompt token count must stay stable"
        );
    }
    wait_log_count(
        &server,
        "paged SSD prefix cache hit",
        hits_before + concurrency,
        Duration::from_secs(20),
    )
    .await;
    assert_cache_kinds(&cache_dir, expected_kinds);

    std::fs::remove_dir_all(&cache_dir).expect("cleanup prefix cache dir");
}

async fn run_text_restart_persistence_case(
    model_dir: PathBuf,
    expected_kinds: &[&str],
    prompt: &str,
) {
    let cache_dir = unique_temp_dir("text-prefix-restart");
    std::fs::create_dir_all(&cache_dir).expect("create prefix cache dir");
    let client = client();
    let body = text_body(prompt);

    let warm_prompt_tokens = {
        let port = alloc_port().await;
        let mut server = ServerProcess::spawn(&model_dir, &cache_dir, port);
        wait_ready(&client, port, &mut server).await;
        let warm = post_chat(&client, port, body.clone()).await;
        assert_cache_kinds(&cache_dir, expected_kinds);
        warm["usage"]["prompt_tokens"].clone()
    };

    let port = alloc_port().await;
    let mut server = ServerProcess::spawn(&model_dir, &cache_dir, port);
    wait_ready(&client, port, &mut server).await;
    let hit = post_chat(&client, port, body).await;
    assert_eq!(hit["usage"]["prompt_tokens"], warm_prompt_tokens);
    wait_log_count(
        &server,
        "paged SSD prefix cache hit",
        1,
        Duration::from_secs(10),
    )
    .await;
    assert_cache_kinds(&cache_dir, expected_kinds);

    std::fs::remove_dir_all(&cache_dir).expect("cleanup prefix cache dir");
}

async fn run_vl_exact_hit_and_image_miss_case(
    label: &str,
    model_dir: PathBuf,
    expected_kinds: &[&str],
) {
    let cache_dir = unique_temp_dir(&format!("{label}-vl-prefix"));
    let image_dir = unique_temp_dir(&format!("{label}-vl-prefix-images"));
    std::fs::create_dir_all(&cache_dir).expect("create prefix cache dir");
    std::fs::create_dir_all(&image_dir).expect("create image temp dir");
    let coco = coco_path();
    let flipped = image_dir.join("coco_flipped.jpg");
    write_flipped_image(&coco, &flipped);

    let port = alloc_port().await;
    let mut server = ServerProcess::spawn(&model_dir, &cache_dir, port);
    let client = client();
    wait_ready(&client, port, &mut server).await;

    let body = vl_body(&coco);
    let warm = post_chat(&client, port, body.clone()).await;
    assert_cache_kinds(&cache_dir, expected_kinds);
    assert_has_vl_fingerprint(&cache_dir);

    let (hit_a, hit_b) = tokio::join!(
        post_chat(&client, port, body.clone()),
        post_chat(&client, port, body.clone())
    );
    assert_eq!(
        hit_a["usage"]["prompt_tokens"],
        warm["usage"]["prompt_tokens"]
    );
    assert_eq!(
        hit_b["usage"]["prompt_tokens"],
        warm["usage"]["prompt_tokens"]
    );
    wait_log_count(
        &server,
        "paged SSD prefix cache hit",
        2,
        Duration::from_secs(10),
    )
    .await;
    let hits_before_negative = count_log(&server.stderr_text(), "paged SSD prefix cache hit");

    let negative = post_chat(&client, port, vl_body(&flipped)).await;
    assert_eq!(
        negative["usage"]["prompt_tokens"], warm["usage"]["prompt_tokens"],
        "flipped image keeps the same prompt shape, so a miss proves image fingerprinting"
    );
    let final_logs = server.stderr_text();
    assert_eq!(
        count_log(&final_logs, "paged SSD prefix cache hit"),
        hits_before_negative,
        "different image must not hit an existing VL prefix cache entry; stderr:\n{final_logs}"
    );
    assert_has_vl_fingerprint(&cache_dir);

    std::fs::remove_dir_all(&cache_dir).expect("cleanup prefix cache dir");
    std::fs::remove_dir_all(&image_dir).expect("cleanup image temp dir");
}

async fn run_vl_concurrent_exact_hit_case(
    label: &str,
    model_dir: PathBuf,
    expected_kinds: &[&str],
    concurrency: usize,
) {
    let cache_dir = unique_temp_dir(&format!("{label}-vl-prefix-concurrent"));
    std::fs::create_dir_all(&cache_dir).expect("create prefix cache dir");

    let port = alloc_port().await;
    let mut server =
        ServerProcess::spawn_with_max_sequences(&model_dir, &cache_dir, port, concurrency);
    let client = client();
    wait_ready(&client, port, &mut server).await;

    let body = vl_body(&coco_path());
    let warm = post_chat(&client, port, body.clone()).await;
    assert_cache_kinds(&cache_dir, expected_kinds);
    assert_has_vl_fingerprint(&cache_dir);

    let hits_before = count_log(&server.stderr_text(), "paged SSD prefix cache hit");
    let responses = post_same_body_concurrently(&client, port, body, concurrency).await;
    for response in responses {
        assert_eq!(
            response["usage"]["prompt_tokens"], warm["usage"]["prompt_tokens"],
            "concurrent VL exact-hit prompt token count must stay stable"
        );
    }
    wait_log_count(
        &server,
        "paged SSD prefix cache hit",
        hits_before + concurrency,
        Duration::from_secs(30),
    )
    .await;
    assert_cache_kinds(cache_dir.as_path(), expected_kinds);
    assert_has_vl_fingerprint(&cache_dir);

    std::fs::remove_dir_all(&cache_dir).expect("cleanup prefix cache dir");
}

async fn run_turboquant_paged_prefix_text_and_vl_case(kv_quant: &str) {
    let cache_dir = unique_temp_dir(&format!("gemma4-moe-{kv_quant}-paged-prefix"));
    std::fs::create_dir_all(&cache_dir).expect("create prefix cache dir");

    let port = alloc_port().await;
    let mut server =
        ServerProcess::spawn_with_kv_quant(&gemma4_moe_model_dir(), &cache_dir, kv_quant, port, 2);
    let client = client();
    wait_ready(&client, port, &mut server).await;

    let text_body = text_body(&format!(
        "For Gemma4 MoE {kv_quant} paged prefix validation, answer briefly."
    ));
    let text_warm = post_chat(&client, port, text_body.clone()).await;
    assert_cache_kinds(&cache_dir, &["full_turbo_quant_packed"]);

    let text_hits_before = count_log(&server.stderr_text(), "paged SSD prefix cache hit");
    let text_hit = post_chat(&client, port, text_body).await;
    assert_eq!(
        text_hit["usage"]["prompt_tokens"], text_warm["usage"]["prompt_tokens"],
        "{kv_quant} text exact-hit prompt token count must stay stable"
    );
    wait_log_count(
        &server,
        "paged SSD prefix cache hit",
        text_hits_before + 1,
        Duration::from_secs(20),
    )
    .await;
    assert_cache_kinds(&cache_dir, &["full_turbo_quant_packed"]);

    let vl_body = vl_body(&coco_path());
    let vl_warm = post_chat(&client, port, vl_body.clone()).await;
    assert_cache_kinds(&cache_dir, &["full_turbo_quant_packed"]);
    assert_has_vl_fingerprint(&cache_dir);

    let vl_hits_before = count_log(&server.stderr_text(), "paged SSD prefix cache hit");
    let vl_hit = post_chat(&client, port, vl_body).await;
    assert_eq!(
        vl_hit["usage"]["prompt_tokens"], vl_warm["usage"]["prompt_tokens"],
        "{kv_quant} VL exact-hit prompt token count must stay stable"
    );
    wait_log_count(
        &server,
        "paged SSD prefix cache hit",
        vl_hits_before + 1,
        Duration::from_secs(30),
    )
    .await;
    assert_cache_kinds(&cache_dir, &["full_turbo_quant_packed"]);
    assert_has_vl_fingerprint(&cache_dir);

    std::fs::remove_dir_all(&cache_dir).expect("cleanup prefix cache dir");
}

async fn run_turboquant_paged_prefix_active_kv_text_and_vl_case(kv_quant: &str) {
    let cache_dir = unique_temp_dir(&format!("gemma4-moe-{kv_quant}-active-kv-prefix"));
    let active_kv_dir = unique_temp_dir(&format!("gemma4-moe-{kv_quant}-active-kv-offload"));
    std::fs::create_dir_all(&cache_dir).expect("create prefix cache dir");

    let port = alloc_port().await;
    let mut server = ServerProcess::spawn_with_kv_quant_and_active_kv(
        &gemma4_moe_model_dir(),
        &cache_dir,
        kv_quant,
        &active_kv_dir,
        port,
        2,
    );
    let client = client();
    wait_ready(&client, port, &mut server).await;
    assert_active_kv_health(&healthz(&client, port).await);

    let text_body = text_body(&format!(
        "For Gemma4 MoE {kv_quant} paged prefix and Active KV validation, answer briefly."
    ));
    let text_warm = post_chat(&client, port, text_body.clone()).await;
    assert_cache_kinds(&cache_dir, &["full_turbo_quant_packed"]);

    let text_hits_before = count_log(&server.stderr_text(), "paged SSD prefix cache hit");
    let text_hit = post_chat(&client, port, text_body).await;
    assert_eq!(
        text_hit["usage"]["prompt_tokens"], text_warm["usage"]["prompt_tokens"],
        "{kv_quant} text exact-hit prompt token count must stay stable"
    );
    wait_log_count(
        &server,
        "paged SSD prefix cache hit",
        text_hits_before + 1,
        Duration::from_secs(20),
    )
    .await;
    assert_cache_kinds(&cache_dir, &["full_turbo_quant_packed"]);

    let vl_body = vl_body(&coco_path());
    let vl_warm = post_chat(&client, port, vl_body.clone()).await;
    assert_cache_kinds(&cache_dir, &["full_turbo_quant_packed"]);
    assert_has_vl_fingerprint(&cache_dir);

    let vl_hits_before = count_log(&server.stderr_text(), "paged SSD prefix cache hit");
    let vl_hit = post_chat(&client, port, vl_body).await;
    assert_eq!(
        vl_hit["usage"]["prompt_tokens"], vl_warm["usage"]["prompt_tokens"],
        "{kv_quant} VL exact-hit prompt token count must stay stable"
    );
    wait_log_count(
        &server,
        "paged SSD prefix cache hit",
        vl_hits_before + 1,
        Duration::from_secs(30),
    )
    .await;
    assert_cache_kinds(&cache_dir, &["full_turbo_quant_packed"]);
    assert_has_vl_fingerprint(&cache_dir);

    let after = healthz(&client, port).await;
    assert_active_kv_health(&after);
    assert_eq!(
        after["active_kv_offload"]["parked_requests"].as_u64(),
        Some(0),
        "HTTP TurboQuant+Paged Prefix+Active KV should not leak parked requests: {after}"
    );

    std::fs::remove_dir_all(&cache_dir).expect("cleanup prefix cache dir");
    std::fs::remove_dir_all(&active_kv_dir).ok();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires QWEN35_MODEL/MLX_DIR or default local Qwen3.5 checkpoint"]
async fn qwen35_text_linear_paged_prefix_cache_batched_exact_hit() {
    run_text_exact_hit_case(
        qwen35_model_dir(),
        &["full_paged", "linear"],
        "For prefix cache validation, answer with one concise sentence about deterministic reuse.",
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires QWEN36_DENSE_MODEL/MLX_DIR or default local Qwen3.6-27B checkpoint"]
async fn qwen36_dense_active_kv_offload_restores_gated_delta_linear_state() {
    let cache_dir = unique_temp_dir("qwen36-dense-active-kv-prefix");
    let active_kv_dir = unique_temp_dir("qwen36-dense-active-kv-offload");
    std::fs::create_dir_all(&cache_dir).expect("create prefix cache dir");

    let port = alloc_port().await;
    let mut server = ServerProcess::spawn_with_active_kv(
        &qwen36_dense_model_dir(),
        &cache_dir,
        &active_kv_dir,
        port,
        2,
    );
    let client = client();
    wait_ready(&client, port, &mut server).await;

    let before = healthz(&client, port).await;
    assert_active_kv_health(&before);
    assert_eq!(
        before["active_kv_offload"]["supported_cache_kinds"],
        serde_json::json!([
            "full_attention_dense",
            "full_attention_paged",
            "turboquant_full_attention_packed",
            "mla",
            "gated_delta_linear",
            "mtp_speculative_side_cache"
        ]),
        "Qwen3.6 Active KV health must advertise GatedDelta/Linear support: {before}"
    );

    let body = serde_json::json!({
        "model": "paged-prefix-matrix",
        "messages": [{
            "role": "user",
            "content": "Continue with a concise numbered sequence and do not stop before ten items."
        }],
        "max_tokens": 32,
        "temperature": 0.0,
        "stream": false
    });
    let responses = post_same_body_concurrently(&client, port, body, 3).await;
    assert_eq!(responses.len(), 3);

    let after = healthz(&client, port).await;
    assert_active_kv_health(&after);
    assert!(
        after["active_kv_offload"]["swap_out_count"]
            .as_u64()
            .unwrap_or_default()
            >= 1,
        "Qwen3.6 mixed Full + Linear cache should swap out: {after}"
    );
    assert!(
        after["active_kv_offload"]["swap_in_count"]
            .as_u64()
            .unwrap_or_default()
            >= 1,
        "Qwen3.6 mixed Full + Linear cache should swap in: {after}"
    );
    assert_eq!(
        after["active_kv_offload"]["parked_requests"].as_u64(),
        Some(0),
        "Qwen3.6 Active KV should not leak parked requests: {after}"
    );

    drop(server);
    std::fs::remove_dir_all(&cache_dir).expect("cleanup prefix cache dir");
    std::fs::remove_dir_all(&active_kv_dir).ok();
}

async fn run_qwen_dense_mtp_active_kv_offload_case(
    model_name: &str,
    temp_name: &str,
    model_dir: PathBuf,
    mtp_model_dir: PathBuf,
) {
    let cache_dir = unique_temp_dir(&format!("{temp_name}-mtp-active-kv-prefix"));
    let active_kv_dir = unique_temp_dir(&format!("{temp_name}-mtp-active-kv-offload"));
    std::fs::create_dir_all(&cache_dir).expect("create prefix cache dir");

    let port = alloc_port().await;
    let mut server = ServerProcess::spawn_with_mtp_and_active_kv(
        &model_dir,
        &mtp_model_dir,
        &cache_dir,
        &active_kv_dir,
        port,
        2,
    );
    let client = client();
    wait_ready(&client, port, &mut server).await;

    let before = healthz(&client, port).await;
    assert_active_kv_health(&before);
    assert_mtp_draft_width(&before, 2);
    assert_eq!(
        before["active_kv_offload"]["supported_cache_kinds"],
        serde_json::json!([
            "full_attention_dense",
            "full_attention_paged",
            "turboquant_full_attention_packed",
            "mla",
            "gated_delta_linear",
            "mtp_speculative_side_cache"
        ]),
        "{model_name} Active KV health must advertise MTP speculative side-cache support: {before}"
    );

    let body = serde_json::json!({
        "model": "paged-prefix-matrix",
        "messages": [{
            "role": "user",
            "content": "Continue with a concise numbered sequence and do not stop before twenty items."
        }],
        "max_tokens": 64,
        "temperature": 0.0,
        "stream": false
    });
    let responses = post_same_body_concurrently(&client, port, body, 3).await;
    assert_eq!(responses.len(), 3);

    let after = healthz(&client, port).await;
    assert_active_kv_health(&after);
    assert!(
        after["mtp"]["drafted_tokens"].as_u64().unwrap_or_default() > 0,
        "{model_name} requests must exercise MTP drafting: {after}"
    );
    // Concurrent Active-KV scheduling records zero-draft control windows, so
    // aggregate drafted_tokens > windows is not a valid multi-token predicate.
    // The dedicated d=1 versus d=2 parity case verifies multi-token drafting.
    assert!(
        after["mtp"]["accepted_draft_tokens"]
            .as_u64()
            .unwrap_or_default()
            > 0,
        "{model_name} concurrent requests must accept at least one MTP draft: {after}"
    );
    assert!(
        after["active_kv_offload"]["swap_out_count"]
            .as_u64()
            .unwrap_or_default()
            >= 1,
        "{model_name} MTP request should swap out: {after}"
    );
    assert!(
        after["active_kv_offload"]["swap_in_count"]
            .as_u64()
            .unwrap_or_default()
            >= 1,
        "{model_name} MTP request should swap in: {after}"
    );
    assert_eq!(
        after["active_kv_offload"]["parked_requests"].as_u64(),
        Some(0),
        "{model_name} MTP Active KV should not leak parked requests: {after}"
    );

    drop(server);
    std::fs::remove_dir_all(&cache_dir).expect("cleanup prefix cache dir");
    std::fs::remove_dir_all(&active_kv_dir).ok();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires QWEN36_DENSE_MODEL, QWEN36_DENSE_MTP_MODEL, and MLX_DIR pointing to real local checkpoints"]
async fn qwen36_dense_mtp_active_kv_offload_restores_speculative_side_cache() {
    run_qwen_dense_mtp_active_kv_offload_case(
        "Qwen3.6",
        "qwen36-dense",
        qwen36_dense_model_dir(),
        qwen36_dense_mtp_model_dir(),
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires QWEN38_DENSE_MODEL, QWEN38_DENSE_MTP_MODEL, and MLX_DIR pointing to real local checkpoints"]
async fn qwen38_dense_mtp_active_kv_offload_restores_speculative_side_cache() {
    run_qwen_dense_mtp_active_kv_offload_case(
        "Qwen3.8",
        "qwen38-dense",
        qwen38_dense_model_dir(),
        qwen38_dense_mtp_model_dir(),
    )
    .await;
}

async fn run_paged_kv_multi_token_parity_case(
    model_name: &str,
    temp_name: &str,
    model_dir: PathBuf,
    mtp_model_dir: PathBuf,
    prompt: &str,
) {
    let body = serde_json::json!({
        "model": "paged-prefix-matrix",
        "messages": [{"role": "user", "content": prompt}],
        // Keep this parity probe below the adaptive policy's eight d=1 samples.
        // Otherwise later zero-draft control windows can make the aggregate
        // drafted_tokens <= windows even after a real d=2 window ran.
        "max_tokens": 8,
        "temperature": 0.0,
        "stream": false
    });

    let single_cache_dir = unique_temp_dir(&format!("{temp_name}-mtp-paged-d1"));
    std::fs::create_dir_all(&single_cache_dir).expect("create single-token prefix cache dir");
    let single_port = alloc_port().await;
    let mut single_server = ServerProcess::spawn_with_mtp_draft_tokens(
        &model_dir,
        &mtp_model_dir,
        &single_cache_dir,
        single_port,
        1,
    );
    let client = client();
    wait_ready(&client, single_port, &mut single_server).await;
    assert_mtp_draft_width(&healthz(&client, single_port).await, 1);
    let single =
        post_chat_with_server_diagnostics(&client, single_port, body.clone(), &single_server).await;
    drop(single_server);
    std::fs::remove_dir_all(&single_cache_dir).expect("cleanup single-token prefix cache dir");

    let multi_cache_dir = unique_temp_dir(&format!("{temp_name}-mtp-paged-d2"));
    std::fs::create_dir_all(&multi_cache_dir).expect("create multi-token prefix cache dir");
    let multi_port = alloc_port().await;
    let mut multi_server = ServerProcess::spawn_with_mtp_draft_tokens(
        &model_dir,
        &mtp_model_dir,
        &multi_cache_dir,
        multi_port,
        2,
    );
    wait_ready(&client, multi_port, &mut multi_server).await;
    let before = healthz(&client, multi_port).await;
    assert_mtp_draft_width(&before, 2);
    let multi = post_chat_with_server_diagnostics(&client, multi_port, body, &multi_server).await;
    let after = healthz(&client, multi_port).await;

    assert_eq!(
        multi["choices"][0]["message"], single["choices"][0]["message"],
        "{model_name} Paged KV multi-token MTP must preserve the exact generated message"
    );
    assert_eq!(
        multi["choices"][0]["finish_reason"], single["choices"][0]["finish_reason"],
        "{model_name} Paged KV multi-token MTP must preserve the finish reason"
    );
    assert_eq!(
        multi["usage"]["completion_tokens"], single["usage"]["completion_tokens"],
        "{model_name} Paged KV multi-token MTP must preserve the completion length"
    );
    assert!(
        after["mtp"]["drafted_tokens"].as_u64().unwrap_or_default()
            > after["mtp"]["windows"].as_u64().unwrap_or_default(),
        "{model_name} draft=2 must execute at least one multi-token MTP window: before={before}, after={after}"
    );
    assert_eq!(
        after["scheduler"]["b_active"].as_u64(),
        Some(0),
        "{model_name} request must release its scheduler slot: {after}"
    );

    drop(multi_server);
    std::fs::remove_dir_all(&multi_cache_dir).expect("cleanup multi-token prefix cache dir");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires QWEN38_DENSE_MODEL, QWEN38_DENSE_MTP_MODEL, and MLX_DIR pointing to real local checkpoints"]
async fn qwen38_dense_paged_kv_multi_token_matches_single_token_mtp() {
    run_paged_kv_multi_token_parity_case(
        "Qwen3.8",
        "qwen38-dense",
        qwen38_dense_model_dir(),
        qwen38_dense_mtp_model_dir(),
        "In one deterministic paragraph, explain why transaction rollback must restore the exact accepted KV prefix.",
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires GEMMA4_LONG_CONTEXT_MODEL, GEMMA4_LONG_CONTEXT_DRAFTER, and MLX_DIR pointing to real local checkpoints"]
async fn gemma4_unified_paged_kv_multi_token_matches_single_token_drafter() {
    run_paged_kv_multi_token_parity_case(
        "Gemma4 Unified 12B",
        "gemma4-unified-12b",
        gemma4_unified_model_dir(),
        gemma4_unified_drafter_model_dir(),
        "In one deterministic paragraph, explain why an accepted speculative prefix must be committed atomically.",
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires QWEN38_DENSE_MODEL, QWEN38_DENSE_MTP_MODEL, and MLX_DIR pointing to real local checkpoints"]
async fn qwen38_dense_mtp_long_context_remains_on_exact_path() {
    let minimum_context_tokens = std::env::var("MTP_LONG_CONTEXT_TOKENS")
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()
        .expect("parse MTP_LONG_CONTEXT_TOKENS")
        .unwrap_or(8_192);
    assert!(
        minimum_context_tokens > 4_096,
        "long-context MTP acceptance must cross the former 4096-token cap"
    );

    let model_dir = qwen38_dense_model_dir();
    let mtp_model_dir = qwen38_dense_mtp_model_dir();
    let cache_dir = unique_temp_dir("qwen38-dense-mtp-long-context-prefix");
    std::fs::create_dir_all(&cache_dir).expect("create prefix cache dir");
    let max_cache_cap = minimum_context_tokens.saturating_add(4_096);
    let port = alloc_port().await;
    let mut server = ServerProcess::spawn_with_mtp_long_context(
        &model_dir,
        &mtp_model_dir,
        &cache_dir,
        port,
        max_cache_cap,
    );
    let client = long_context_client();
    wait_ready(&client, port, &mut server).await;

    let before = healthz(&client, port).await;
    assert_mtp_draft_width(&before, 2);
    let body = serde_json::json!({
        "model": "paged-prefix-matrix",
        "messages": [{
            "role": "user",
            "content": long_context_prompt(&model_dir, minimum_context_tokens)
        }],
        "max_tokens": 32,
        "temperature": 0.0,
        "stream": false
    });
    let response = post_chat_with_server_diagnostics(&client, port, body, &server).await;
    assert!(
        response["usage"]["prompt_tokens"]
            .as_u64()
            .unwrap_or_default()
            >= minimum_context_tokens as u64,
        "request did not reach the requested long context: {response}"
    );

    let after = healthz(&client, port).await;
    for field in [
        "prefill_count",
        "step_count",
        "drafted_tokens",
        "accepted_draft_tokens",
    ] {
        assert!(
            after["mtp"][field].as_u64().unwrap_or_default()
                > before["mtp"][field].as_u64().unwrap_or_default(),
            "long-context request must increase mtp.{field}: before={before}, after={after}"
        );
    }
    assert_eq!(
        after["mtp"]["fallback_prefill_count"], before["mtp"]["fallback_prefill_count"],
        "long-context request must not fall back from MTP: before={before}, after={after}"
    );
    assert!(
        after["mtp"]["accepted_draft_tokens"]
            .as_u64()
            .unwrap_or_default()
            <= after["mtp"]["drafted_tokens"].as_u64().unwrap_or_default(),
        "accepted MTP drafts cannot exceed proposed drafts: {after}"
    );
    assert!(
        after["mtp"]["drafted_tokens"].as_u64().unwrap_or_default()
            > after["mtp"]["windows"].as_u64().unwrap_or_default(),
        "long-context Paged KV must execute at least one multi-token MTP window: {after}"
    );
    assert_eq!(
        after["scheduler"]["b_active"].as_u64(),
        Some(0),
        "long-context MTP request must release its scheduler slot: {after}"
    );
    eprintln!(
        "Qwen3.8 MTP long-context acceptance: prompt_tokens={} prefill_count={} step_count={} \
         drafted_tokens={} accepted_draft_tokens={} fallback_prefill_count={}",
        response["usage"]["prompt_tokens"],
        after["mtp"]["prefill_count"],
        after["mtp"]["step_count"],
        after["mtp"]["drafted_tokens"],
        after["mtp"]["accepted_draft_tokens"],
        after["mtp"]["fallback_prefill_count"]
    );

    drop(server);
    std::fs::remove_dir_all(&cache_dir).expect("cleanup prefix cache dir");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires QWEN35_MODEL/MLX_DIR or default local Qwen3.5 checkpoint"]
async fn qwen35_text_paged_prefix_cache_persists_across_server_restart() {
    run_text_restart_persistence_case(
        qwen35_model_dir(),
        &["full_paged", "linear"],
        "For restart persistence validation, answer with one concise sentence about deterministic reuse.",
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires GLM47_MODEL_DIR/MLX_DIR or default local GLM checkpoint"]
async fn glm47_mla_paged_prefix_cache_batched_exact_hit() {
    run_text_exact_hit_case(
        glm47_model_dir(),
        &["mla"],
        "For MLA prefix cache validation, answer with one concise sentence.",
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires QWEN35_MODEL/MLX_DIR or default local Qwen3.5 checkpoint"]
async fn qwen35_vl_paged_prefix_cache_exact_hit_and_image_miss_without_mtp() {
    run_vl_exact_hit_and_image_miss_case("qwen35", qwen35_model_dir(), &["full_paged", "linear"])
        .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires MINICPMV46_MODEL/MLX_DIR or default local MiniCPM-V checkpoint"]
async fn minicpmv46_vl_paged_prefix_cache_exact_hit_and_image_miss() {
    run_vl_exact_hit_and_image_miss_case(
        "minicpmv46",
        minicpmv46_model_dir(),
        &["full_paged", "linear"],
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires GEMMA4_MODEL/MLX_DIR or default local Gemma4 checkpoint"]
async fn gemma4_vl_paged_prefix_cache_exact_hit_and_image_miss() {
    run_vl_exact_hit_and_image_miss_case("gemma4", gemma4_model_dir(), &["full_paged"]).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires GEMMA4_MOE_MODEL/MLX_DIR or default local Gemma4 MoE checkpoint"]
async fn gemma4_moe_text_paged_prefix_cache_batched_exact_hit() {
    run_text_exact_hit_case(
        gemma4_moe_model_dir(),
        &["full_paged"],
        "For Gemma4 MoE prefix cache validation, answer with one concise sentence.",
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires GEMMA4_MOE_MODEL/MLX_DIR or default local Gemma4 MoE checkpoint"]
async fn gemma4_moe_text_paged_prefix_cache_c4_exact_hit() {
    run_text_concurrent_exact_hit_case(
        gemma4_moe_model_dir(),
        &["full_paged"],
        "For Gemma4 MoE concurrent prefix cache validation, answer with one concise sentence.",
        4,
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires GEMMA4_MOE_MODEL/MLX_DIR or default local Gemma4 MoE checkpoint"]
async fn gemma4_moe_vl_paged_prefix_cache_exact_hit_and_image_miss() {
    run_vl_exact_hit_and_image_miss_case("gemma4_moe", gemma4_moe_model_dir(), &["full_paged"])
        .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires GEMMA4_MOE_MODEL/MLX_DIR or default local Gemma4 MoE checkpoint"]
async fn gemma4_moe_vl_paged_prefix_cache_c4_exact_hit() {
    run_vl_concurrent_exact_hit_case("gemma4_moe", gemma4_moe_model_dir(), &["full_paged"], 4)
        .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires GEMMA4_MOE_MODEL/MLX_DIR or default local Gemma4 MoE checkpoint"]
async fn gemma4_moe_turboquant_kv_paged_prefix_cache_text_and_vl_exact_hit() {
    for kv_quant in ["turbo3", "turbo4", "k3v4"] {
        run_turboquant_paged_prefix_text_and_vl_case(kv_quant).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires GEMMA4_MOE_MODEL/MLX_DIR or default local Gemma4 MoE checkpoint"]
async fn gemma4_moe_turboquant_kv_paged_prefix_cache_active_kv_http_text_and_vl() {
    for kv_quant in ["turbo3", "turbo4", "k3v4"] {
        run_turboquant_paged_prefix_active_kv_text_and_vl_case(kv_quant).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires GEMMA4_MOE_MODEL/MLX_DIR or default local Gemma4 MoE checkpoint"]
async fn gemma4_moe_active_kv_offload_http_text_and_vl_health() {
    let cache_dir = unique_temp_dir("gemma4-moe-active-kv-http-prefix");
    let active_kv_dir = unique_temp_dir("gemma4-moe-active-kv-http-offload");
    std::fs::create_dir_all(&cache_dir).expect("create prefix cache dir");

    let port = alloc_port().await;
    let mut server = ServerProcess::spawn_with_active_kv(
        &gemma4_moe_model_dir(),
        &cache_dir,
        &active_kv_dir,
        port,
        1,
    );
    let client = client();
    wait_ready(&client, port, &mut server).await;
    assert_active_kv_health(&healthz(&client, port).await);

    post_chat(
        &client,
        port,
        text_body("For Gemma4 MoE Active KV health validation, answer briefly."),
    )
    .await;
    post_chat(&client, port, vl_body(&coco_path())).await;

    let after = healthz(&client, port).await;
    assert_active_kv_health(&after);
    assert_eq!(
        after["active_kv_offload"]["parked_requests"].as_u64(),
        Some(0),
        "HTTP Active KV should not leak parked requests: {after}"
    );

    std::fs::remove_dir_all(&cache_dir).expect("cleanup prefix cache dir");
    std::fs::remove_dir_all(&active_kv_dir).ok();
}

#[test]
#[ignore = "requires GEMMA4_MODEL/MLX_DIR or default local Gemma4 checkpoint"]
fn gemma4_vl_split_prefill_paged_kv_matches_dense_argmax() {
    let model_dir = gemma4_model_dir();
    let loader = Loader::open_multimodal(&model_dir).expect("open Gemma4 loader");
    let tokenizer = Tokenizer::from_loader(&loader).expect("Gemma4 tokenizer");
    let cfg = ironmlx::models::Gemma4Config::from_loader(&loader).expect("Gemma4 config");
    let vision = VisionInputConfig::Gemma4 {
        vision_config: cfg
            .vision_config
            .clone()
            .expect("Gemma4 checkpoint must include vision config"),
    };
    let model = ironmlx::models::Gemma4Model::from_loader(&loader).expect("Gemma4 model");
    let (prompt_ids, pixel_values, grid_thw, image_token_id) =
        gemma4_prompt_inputs(&tokenizer, &vision);

    let dense = gemma4_split_prefill_argmax(
        &model,
        &prompt_ids,
        &pixel_values,
        &grid_thw,
        image_token_id,
        false,
    );
    let paged = gemma4_split_prefill_argmax(
        &model,
        &prompt_ids,
        &pixel_values,
        &grid_thw,
        image_token_id,
        true,
    );
    let eos = tokenizer.eos_token_ids();
    eprintln!(
        "Gemma4 split-prefill argmax dense={dense} paged={paged} eos={eos:?} dense_text={:?} paged_text={:?}",
        tokenizer.decode(&[dense], false).unwrap_or_default(),
        tokenizer.decode(&[paged], false).unwrap_or_default(),
    );
    assert!(
        !eos.contains(&dense),
        "dense split-prefill unexpectedly produced EOS token {dense}"
    );
    assert_eq!(
        paged, dense,
        "Gemma4 paged KV split-prefill must preserve dense greedy argmax"
    );
}
