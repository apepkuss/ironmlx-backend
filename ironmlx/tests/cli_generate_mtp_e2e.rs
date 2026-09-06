//! CLI generate + MTP real-checkpoint smoke tests.
//!
//! These tests are ignored by default because they require local MLX runtime and
//! Hugging Face checkpoint snapshots.

#[path = "common/ironmlx_process.rs"]
mod ironmlx_process;

use std::path::PathBuf;

fn require_env_path(name: &str) -> PathBuf {
    PathBuf::from(std::env::var(name).unwrap_or_else(|_| panic!("{name} must be set")))
}

fn coco_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("qwen35_vl")
        .join("coco_sample.jpg")
}

fn run_text_generate_smoke(model_env: &str, mtp_model_env: &str) {
    let model_dir = require_env_path(model_env);
    let mtp_model_dir = require_env_path(mtp_model_env);
    let output = ironmlx_process::command()
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .arg("generate")
        .arg("--model")
        .arg(&model_dir)
        .arg("--mtp-model-dir")
        .arg(&mtp_model_dir)
        .arg("--mtp-draft-tokens")
        .arg("1")
        .arg("--prompt")
        .arg("Answer with exactly one word: OK")
        .arg("--max-tokens")
        .arg("1")
        .arg("--temperature")
        .arg("0")
        .output()
        .expect("run ironmlx generate");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "generate exited with {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        stdout,
        stderr
    );
    assert!(
        !stdout.trim().is_empty(),
        "generate should emit at least one token; stderr:\n{stderr}"
    );
}

fn run_vl_generate_smoke(model_env: &str, mtp_model_env: &str) {
    let model_dir = require_env_path(model_env);
    let mtp_model_dir = require_env_path(mtp_model_env);
    let output = ironmlx_process::command()
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .arg("generate")
        .arg("--model")
        .arg(&model_dir)
        .arg("--mtp-model-dir")
        .arg(&mtp_model_dir)
        .arg("--mtp-draft-tokens")
        .arg("1")
        .arg("--image")
        .arg(coco_path())
        .arg("--prompt")
        .arg("Describe this image.")
        .arg("--max-tokens")
        .arg("1")
        .arg("--temperature")
        .arg("0")
        .output()
        .expect("run ironmlx generate");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "generate exited with {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        stdout,
        stderr
    );
    assert!(
        !stdout.trim().is_empty(),
        "generate should emit at least one token; stderr:\n{stderr}"
    );
}

#[test]
#[ignore = "requires QWEN35_MODEL, QWEN35_MTP_MODEL, and MLX_DIR pointing to real local checkpoints"]
fn qwen35_text_generate_with_mtp_accepts_request() {
    run_text_generate_smoke("QWEN35_MODEL", "QWEN35_MTP_MODEL");
}

#[test]
#[ignore = "requires QWEN36_DENSE_MODEL, QWEN36_DENSE_MTP_MODEL, and MLX_DIR pointing to real local checkpoints"]
fn qwen36_dense_text_generate_with_mtp_accepts_request() {
    run_text_generate_smoke("QWEN36_DENSE_MODEL", "QWEN36_DENSE_MTP_MODEL");
}

#[test]
#[ignore = "requires QWEN38_DENSE_MODEL, QWEN38_DENSE_MTP_MODEL, and MLX_DIR pointing to real local checkpoints"]
fn qwen38_dense_text_generate_with_mtp_accepts_request() {
    run_text_generate_smoke("QWEN38_DENSE_MODEL", "QWEN38_DENSE_MTP_MODEL");
}

#[test]
#[ignore = "requires QWEN38_DENSE_MODEL, QWEN38_DENSE_MTP_MODEL, and MLX_DIR pointing to real local checkpoints"]
fn qwen38_dense_vl_generate_with_mtp_accepts_image_request() {
    run_vl_generate_smoke("QWEN38_DENSE_MODEL", "QWEN38_DENSE_MTP_MODEL");
}

#[test]
#[ignore = "requires QWEN36_MOE_MODEL, QWEN36_MOE_MTP_MODEL, and MLX_DIR pointing to real local checkpoints"]
fn qwen36_moe_text_generate_with_mtp_accepts_request() {
    run_text_generate_smoke("QWEN36_MOE_MODEL", "QWEN36_MOE_MTP_MODEL");
}

#[test]
#[ignore = "requires QWEN35_MODEL, QWEN35_MTP_MODEL, and MLX_DIR pointing to real local checkpoints"]
fn qwen35_vl_generate_with_mtp_accepts_image_request() {
    run_vl_generate_smoke("QWEN35_MODEL", "QWEN35_MTP_MODEL");
}

#[test]
#[ignore = "requires QWEN36_MOE_MODEL, QWEN36_MOE_MTP_MODEL, and MLX_DIR pointing to real local checkpoints"]
fn qwen36_moe_vl_generate_with_mtp_accepts_image_request() {
    run_vl_generate_smoke("QWEN36_MOE_MODEL", "QWEN36_MOE_MTP_MODEL");
}
