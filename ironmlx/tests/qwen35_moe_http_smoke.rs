//! P5c HTTP smoke: launch ironmlx serve with Qwen35MoeModel, post
//! a chat completion, verify SSE stream completes with valid token output.
//!
//! Run with:
//!   IRONMLX_MOE_MODEL_DIR=<snapshot> MLX_DIR=$HOME/.local/mlx \
//!     cargo test -p ironmlx --release --test qwen35_moe_http_smoke \
//!       -- --ignored --nocapture --test-threads=1
//!
//! Time budget: ~30-60s for model load + a few seconds per token decode.

#[path = "common/ironmlx_process.rs"]
mod ironmlx_process;

use std::process::Stdio;
use std::time::Duration;

fn locate_snapshot() -> String {
    if let Ok(p) = std::env::var("IRONMLX_MOE_MODEL_DIR") {
        return p;
    }
    let home = std::env::var("HOME").expect("HOME env");
    let glob =
        format!("{home}/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots");
    let entries = std::fs::read_dir(&glob).expect("snapshots dir");
    let first = entries
        .filter_map(|e| e.ok())
        .next()
        .expect("at least one snapshot");
    first.path().to_string_lossy().into_owned()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn p5c_http_smoke_chat_completion_non_stream() {
    let dir = locate_snapshot();

    // Pick an unused port (bind 0 to let OS choose).
    let port = {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let p = l.local_addr().unwrap().port();
        drop(l);
        p
    };

    // Spawn ironmlx serve with the metallib installed alongside the linked MLX.
    let mut cmd = ironmlx_process::command();
    cmd.args([
        "serve",
        "--model",
        &dir,
        "--port",
        &port.to_string(),
        "--b-max",
        "1",
        "--max-cache-cap",
        "4096",
    ])
    .stdout(Stdio::piped())
    .stderr(Stdio::piped());
    let mut child = cmd.spawn().expect("spawn ironmlx serve");

    // Wait up to 120s for /healthz 200 (MoE load + warmup is heavier than dense).
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .no_proxy()
        .build()
        .unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut up = false;
    for i in 0..120 {
        tokio::time::sleep(Duration::from_secs(1)).await;
        match client.get(format!("{base}/healthz")).send().await {
            Ok(r) if r.status().is_success() => {
                up = true;
                eprintln!("serve healthy after {}s", i + 1);
                break;
            }
            _ => {}
        }
    }
    if !up {
        // Capture stderr before killing, for diagnostics.
        child.kill().ok();
        let output = child.wait_with_output().ok();
        if let Some(out) = output {
            eprintln!(
                "child stderr: {}",
                String::from_utf8_lossy(&out.stderr)
                    .chars()
                    .take(4096)
                    .collect::<String>()
            );
        }
        panic!("serve did not become healthy within 120s");
    }

    // Post a tiny chat completion (non-streaming).
    let body = serde_json::json!({
        "model": "qwen3_5_moe",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 5,
        "temperature": 0.0,
        "stream": false,
    });
    let resp = client
        .post(format!("{base}/v1/chat/completions"))
        .timeout(Duration::from_secs(120))
        .json(&body)
        .send()
        .await
        .expect("post /v1/chat/completions");

    let status = resp.status();
    let body_text = resp.text().await.expect("body");
    eprintln!("HTTP {} body: {}", status, body_text);
    assert!(
        status.is_success(),
        "HTTP {} from /v1/chat/completions (body: {})",
        status,
        body_text
    );

    let v: serde_json::Value = serde_json::from_str(&body_text).expect("json");
    let content = v["choices"][0]["message"]["content"].as_str().unwrap_or("");
    assert!(
        !content.is_empty(),
        "empty completion content: {}",
        body_text
    );
    eprintln!("got content: {:?}", content);

    // Graceful shutdown.
    child.kill().ok();
    child.wait().ok();
}
