//! Real-checkpoint E2E for Qwen VL + MTP + paged SSD prefix cache.
//!
//! Requires real local checkpoint snapshot dirs:
//!
//! ```text
//! MLX_DIR=$HOME/.local/mlx \
//! QWEN35_MODEL=/path/to/Qwen3.5-4B-MLX-4bit/snapshots/<sha> \
//! QWEN35_MTP_MODEL=/path/to/Qwen3.5-4B-MTP-4bit/snapshots/<sha> \
//! cargo test --release -p ironmlx --test vl_mtp_paged_prefix_e2e \
//!   -- --ignored --test-threads=1 --nocapture
//! ```
//!
//! The Qwen3.6 MoE variant uses `QWEN36_MOE_MODEL` and
//! `QWEN36_MOE_MTP_MODEL`.

#[path = "common/ironmlx_process.rs"]
mod ironmlx_process;

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Stdio};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use base64::Engine;

struct ServerProcess {
    child: Child,
    stderr: Arc<Mutex<Vec<u8>>>,
    drainer: Option<JoinHandle<()>>,
}

impl ServerProcess {
    fn spawn(model_dir: &Path, mtp_model_dir: &Path, cache_dir: &Path, port: u16) -> Self {
        let mut cmd = ironmlx_process::command();
        cmd.current_dir(env!("CARGO_MANIFEST_DIR"));
        cmd.args([
            "serve",
            "--model",
            model_dir
                .to_str()
                .expect("QWEN35_MODEL path must be valid UTF-8"),
            "--mtp-model-dir",
            mtp_model_dir
                .to_str()
                .expect("QWEN35_MTP_MODEL path must be valid UTF-8"),
            "--mtp-draft-tokens",
            "1",
            "--paged-prefix-cache-dir",
            cache_dir
                .to_str()
                .expect("prefix cache path must be valid UTF-8"),
            "--paged-prefix-cache-block-size",
            "16",
            "--paged-prefix-cache-max-pages",
            "4096",
            "--host",
            "127.0.0.1",
            "--port",
            &port.to_string(),
            "--b-max",
            "2",
            "--admission-deadline-ms",
            "200",
            "--admission-queue-max",
            "8",
            "--prefill-chunk-size",
            "0",
            "--max-cache-cap",
            "4096",
        ]);
        cmd.env("RUST_LOG", "ironmlx=debug,warn");
        cmd.env("IRONMLX_CHUNKED_ROLLING_PROFILE", "1");
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

fn openai_vl_body(image_path: &Path) -> serde_json::Value {
    serde_json::json!({
        "model": "qwen-vl-mtp-prefix-e2e",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in one short sentence."},
                {"type": "image_url", "image_url": {"url": image_data_url(image_path)}}
            ]
        }],
        "max_tokens": 4,
        "stream": false
    })
}

fn write_flipped_image(src: &Path, dst: &Path) {
    let img = image::open(src).expect("open source image");
    let flipped =
        image::DynamicImage::ImageRgba8(image::imageops::flip_horizontal(&img.to_rgba8()));
    flipped
        .save_with_format(dst, image::ImageFormat::Jpeg)
        .expect("write flipped image");
}

fn client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(300))
        .no_proxy()
        .build()
        .expect("reqwest client")
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
        json["usage"]["prompt_tokens"].as_u64().unwrap_or_default() >= 64,
        "prompt token count must prove the image path was used: {json}"
    );
    json
}

async fn healthz(client: &reqwest::Client, port: u16) -> serde_json::Value {
    client
        .get(format!("http://127.0.0.1:{port}/healthz"))
        .send()
        .await
        .expect("healthz send")
        .json()
        .await
        .expect("healthz json")
}

fn count_log(text: &str, needle: &str) -> usize {
    text.match_indices(needle).count()
}

fn count_log_lines_with_all(text: &str, needles: &[&str]) -> usize {
    text.lines()
        .filter(|line| needles.iter().all(|needle| line.contains(needle)))
        .count()
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

async fn wait_log_line_with_all(
    server: &ServerProcess,
    needles: &[&str],
    min_count: usize,
    timeout: Duration,
) {
    let deadline = Instant::now() + timeout;
    loop {
        let text = server.stderr_text();
        if count_log_lines_with_all(&text, needles) >= min_count {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "log needles {needles:?} did not reach {min_count}; stderr:\n{text}"
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

fn assert_cache_entries_have_mtp_payload(cache_dir: &Path) {
    let entries = cache_entry_dirs(cache_dir);
    assert!(
        entries.len() >= 2,
        "expected prefix cache entries in {}",
        cache_dir.display()
    );
    for entry in entries {
        let meta_path = entry.join("meta.json");
        let meta: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&meta_path).expect("read cache meta"))
                .expect("cache meta json");
        assert!(
            meta["main_layers"].as_array().map_or(0, Vec::len) > 0,
            "cache entry missing main_layers: {}",
            meta_path.display()
        );
        assert!(
            meta["mtp_layers"].as_array().map_or(0, Vec::len) > 0,
            "cache entry missing mtp_layers: {}",
            meta_path.display()
        );
        assert!(
            !meta["mtp_last_hidden"].is_null(),
            "cache entry missing mtp_last_hidden: {}",
            meta_path.display()
        );
    }
}

async fn run_vl_mtp_paged_prefix_cache_exact_hit_batch_and_image_miss(
    model_env: &str,
    mtp_model_env: &str,
    temp_suffix: &str,
) {
    let model_dir = PathBuf::from(
        std::env::var(model_env).unwrap_or_else(|_| panic!("{model_env} must be set")),
    );
    let mtp_model_dir = PathBuf::from(
        std::env::var(mtp_model_env).unwrap_or_else(|_| panic!("{mtp_model_env} must be set")),
    );
    let cache_dir = unique_temp_dir(&format!("vl-mtp-prefix-cache-{temp_suffix}"));
    let image_dir = unique_temp_dir(&format!("vl-mtp-prefix-images-{temp_suffix}"));
    std::fs::create_dir_all(&cache_dir).expect("create prefix cache dir");
    std::fs::create_dir_all(&image_dir).expect("create image temp dir");
    let coco = coco_path();
    let flipped = image_dir.join("coco_flipped.jpg");
    write_flipped_image(&coco, &flipped);

    let port = alloc_port().await;
    let mut server = ServerProcess::spawn(&model_dir, &mtp_model_dir, &cache_dir, port);
    let client = client();
    wait_ready(&client, port, &mut server).await;

    let initial_health = healthz(&client, port).await;
    assert_eq!(initial_health["mtp"]["enabled"], true, "{initial_health}");

    let body = openai_vl_body(&coco);
    let warm = post_chat(&client, port, body.clone()).await;
    let warm_health = healthz(&client, port).await;
    assert_eq!(warm_health["mtp"]["prefill_count"], 1, "{warm_health}");
    assert_cache_entries_have_mtp_payload(&cache_dir);

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
    let batch_health = healthz(&client, port).await;
    assert_eq!(
        batch_health["mtp"]["prefill_count"], 2,
        "two concurrent exact-hit requests should share one MTP batch prefill: {batch_health}"
    );
    wait_log_count(
        &server,
        "paged SSD prefix cache MTP hit",
        2,
        Duration::from_secs(10),
    )
    .await;
    wait_log_line_with_all(
        &server,
        &["event=fresh_prefill", "active_count=2"],
        1,
        Duration::from_secs(10),
    )
    .await;
    let hits_before_negative = count_log(&server.stderr_text(), "paged SSD prefix cache MTP hit");

    let negative = post_chat(&client, port, openai_vl_body(&flipped)).await;
    assert_eq!(
        negative["usage"]["prompt_tokens"], warm["usage"]["prompt_tokens"],
        "flipped image keeps the same prompt shape, so a miss proves image fingerprinting"
    );
    let negative_health = healthz(&client, port).await;
    assert_eq!(
        negative_health["mtp"]["prefill_count"], 3,
        "{negative_health}"
    );
    wait_log_count(
        &server,
        "paged SSD prefix cache MTP saved",
        6,
        Duration::from_secs(10),
    )
    .await;
    let final_logs = server.stderr_text();
    assert_eq!(
        count_log(&final_logs, "paged SSD prefix cache MTP hit"),
        hits_before_negative,
        "different image must not hit an existing VL prefix cache entry; stderr:\n{final_logs}"
    );
    assert!(
        cache_entry_dirs(&cache_dir).len() >= 4,
        "different-image miss should save distinct prefix cache entries"
    );
    assert_cache_entries_have_mtp_payload(&cache_dir);

    std::fs::remove_dir_all(&cache_dir).expect("cleanup prefix cache dir");
    std::fs::remove_dir_all(&image_dir).expect("cleanup image temp dir");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires QWEN35_MODEL, QWEN35_MTP_MODEL, and MLX_DIR pointing to real local checkpoints"]
async fn qwen35_vl_mtp_paged_prefix_cache_exact_hit_batch_and_image_miss() {
    run_vl_mtp_paged_prefix_cache_exact_hit_batch_and_image_miss(
        "QWEN35_MODEL",
        "QWEN35_MTP_MODEL",
        "qwen35",
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires QWEN36_MOE_MODEL, QWEN36_MOE_MTP_MODEL, and MLX_DIR pointing to real local checkpoints"]
async fn qwen36_moe_vl_mtp_paged_prefix_cache_exact_hit_batch_and_image_miss() {
    run_vl_mtp_paged_prefix_cache_exact_hit_batch_and_image_miss(
        "QWEN36_MOE_MODEL",
        "QWEN36_MOE_MTP_MODEL",
        "qwen36-moe",
    )
    .await;
}
