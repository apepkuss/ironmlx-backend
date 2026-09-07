use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::fd::AsRawFd;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{bail, Context};
use clap::Args;
use hf_hub::api::tokio::{ApiBuilder, Progress};
use hf_hub::{Repo, RepoType};
use reqwest::header::{
    HeaderMap, HeaderName, HeaderValue, AUTHORIZATION, CONTENT_LENGTH, CONTENT_RANGE, ETAG,
    LOCATION, RANGE, USER_AGENT,
};
use reqwest::redirect::Policy;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::Result;

const TRANSFER_IDENTITY_VERSION: u32 = 1;
const HF_PARTIAL_EXTENSION: &str = "sync.part";
const HASH_POLL_INTERVAL: Duration = Duration::from_millis(50);
const PROGRESS_MIN_BYTES: u64 = 4 * 1024 * 1024;
const PROGRESS_MAX_INTERVAL: Duration = Duration::from_millis(200);

#[derive(Args, Debug)]
pub struct HfTransferArgs {
    /// Hugging Face model repository, in organization/model form.
    #[arg(long)]
    repo_id: String,
    /// Immutable Hugging Face commit SHA.
    #[arg(long)]
    revision: String,
    /// Repository-relative file path.
    #[arg(long)]
    filename: String,
    /// Final file location inside the unpublished IronMLX staging snapshot.
    #[arg(long)]
    destination: PathBuf,
    /// Private cache directory used for this file's resumable hf-hub transfer.
    #[arg(long)]
    cache_dir: PathBuf,
    /// Expected file size from the immutable repository manifest.
    #[arg(long)]
    expected_size: u64,
    /// Expected SHA-256 from the immutable repository manifest.
    #[arg(long)]
    expected_sha256: String,
    /// Hugging Face endpoint.
    #[arg(long, default_value = "https://huggingface.co")]
    endpoint: String,
    /// Maximum number of concurrent hf-hub Range chunks.
    #[arg(long, default_value_t = 4)]
    parallelism: usize,
    /// Size of each hf-hub Range chunk.
    #[arg(long, default_value_t = 10_000_000)]
    chunk_size: usize,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct TransferIdentity {
    version: u32,
    provider: String,
    repo_id: String,
    commit_sha: String,
    path: String,
    expected_size: u64,
    expected_sha256: String,
    etag: String,
}

#[derive(Debug)]
struct RemoteFileIdentity {
    commit_sha: String,
    size: u64,
    etag: String,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum TransferEvent<'a> {
    Progress {
        bytes: u64,
        total: u64,
    },
    Complete {
        path: &'a Path,
        size: u64,
        sha256: &'a str,
        etag: &'a str,
    },
}

#[derive(Clone, Default)]
struct JsonProgress {
    state: Arc<Mutex<JsonProgressState>>,
}

struct JsonProgressState {
    bytes: u64,
    total: u64,
    last_emitted_bytes: u64,
    last_emitted_at: Instant,
    awaiting_baseline: bool,
    #[cfg(test)]
    first_emitted_bytes: Option<u64>,
}

impl Default for JsonProgressState {
    fn default() -> Self {
        Self {
            bytes: 0,
            total: 0,
            last_emitted_bytes: 0,
            last_emitted_at: Instant::now(),
            awaiting_baseline: false,
            #[cfg(test)]
            first_emitted_bytes: None,
        }
    }
}

impl Progress for JsonProgress {
    async fn init(&mut self, size: usize, _filename: &str) {
        if let Ok(mut state) = self.state.lock() {
            state.bytes = 0;
            state.total = size as u64;
            state.last_emitted_bytes = 0;
            state.last_emitted_at = Instant::now();
            state.awaiting_baseline = true;
        }
    }

    async fn update(&mut self, size: usize) {
        if let Ok(mut state) = self.state.lock() {
            // hf-hub reports the committed starting offset as its first update,
            // before reporting newly transferred chunks.
            let baseline = std::mem::take(&mut state.awaiting_baseline);
            state.bytes = if baseline {
                (size as u64).min(state.total)
            } else {
                state.bytes.saturating_add(size as u64).min(state.total)
            };
            let now = Instant::now();
            if baseline
                || state.bytes == state.total
                || state.bytes.saturating_sub(state.last_emitted_bytes) >= PROGRESS_MIN_BYTES
                || now.duration_since(state.last_emitted_at) >= PROGRESS_MAX_INTERVAL
            {
                #[cfg(test)]
                if state.first_emitted_bytes.is_none() {
                    state.first_emitted_bytes = Some(state.bytes);
                }
                emit_event(&TransferEvent::Progress {
                    bytes: state.bytes,
                    total: state.total,
                });
                state.last_emitted_bytes = state.bytes;
                state.last_emitted_at = now;
            }
        }
    }

    async fn finish(&mut self) {
        if let Ok(mut state) = self.state.lock() {
            state.bytes = state.total;
            if state.last_emitted_bytes != state.total {
                #[cfg(test)]
                if state.first_emitted_bytes.is_none() {
                    state.first_emitted_bytes = Some(state.bytes);
                }
                emit_event(&TransferEvent::Progress {
                    bytes: state.bytes,
                    total: state.total,
                });
                state.last_emitted_bytes = state.total;
                state.last_emitted_at = Instant::now();
            }
        }
    }
}

struct TransferLock {
    file: File,
}

impl TransferLock {
    fn acquire(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!("creating transfer lock directory {}", parent.display())
            })?;
        }
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(path)
            .with_context(|| format!("opening transfer lock {}", path.display()))?;
        let result = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if result != 0 {
            bail!(
                "another process is already downloading this Hugging Face file: {}",
                path.display()
            );
        }
        Ok(Self { file })
    }
}

impl Drop for TransferLock {
    fn drop(&mut self) {
        unsafe {
            libc::flock(self.file.as_raw_fd(), libc::LOCK_UN);
        }
    }
}

struct HashCursor {
    digest: Sha256,
    bytes_hashed: u64,
}

pub fn run(args: HfTransferArgs) -> Result<()> {
    validate_args(&args)?;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("creating Hugging Face transfer runtime")?;
    runtime.block_on(run_async(args))
}

async fn run_async(args: HfTransferArgs) -> Result<()> {
    run_async_with_progress(args, JsonProgress::default()).await
}

async fn run_async_with_progress(args: HfTransferArgs, progress: JsonProgress) -> Result<()> {
    let expected_sha256 = args.expected_sha256.to_ascii_lowercase();
    if args.destination.exists() {
        let actual_size = args.destination.metadata()?.len();
        let actual_sha256 = sha256_file(&args.destination)?;
        if actual_size == args.expected_size && actual_sha256 == expected_sha256 {
            emit_event(&TransferEvent::Complete {
                path: &args.destination,
                size: actual_size,
                sha256: &actual_sha256,
                etag: "",
            });
            return Ok(());
        }
        bail!(
            "staging destination already exists with unexpected identity: {}",
            args.destination.display()
        );
    }

    let lock_path = transfer_lock_path(&args.cache_dir)?;
    let _lock = TransferLock::acquire(&lock_path)?;
    let token = std::env::var("HF_TOKEN")
        .ok()
        .filter(|value| !value.trim().is_empty());

    let api = ApiBuilder::new()
        .with_endpoint(args.endpoint.clone())
        .with_cache_dir(args.cache_dir.clone())
        .with_token(token.clone())
        .with_max_files(args.parallelism)
        .with_chunk_size(Some(args.chunk_size))
        .with_progress(false)
        .with_user_agent("ironmlx", env!("CARGO_PKG_VERSION"))
        .build()
        .context("initializing hf-hub client")?;
    let repo = Repo::with_revision(args.repo_id.clone(), RepoType::Model, args.revision.clone());
    let repo_folder = repo.folder_name();
    let api_repo = api.repo(repo);
    let url = api_repo.url(&args.filename);
    let remote = resolve_remote_identity(&url, token.as_deref()).await?;
    if !remote.commit_sha.eq_ignore_ascii_case(&args.revision) {
        bail!(
            "Hugging Face returned commit {} for immutable revision {}",
            remote.commit_sha,
            args.revision
        );
    }
    if remote.size != args.expected_size {
        bail!(
            "Hugging Face returned size {} for {}, expected {}",
            remote.size,
            args.filename,
            args.expected_size
        );
    }

    let identity = TransferIdentity {
        version: TRANSFER_IDENTITY_VERSION,
        provider: "huggingface".to_string(),
        repo_id: args.repo_id.clone(),
        commit_sha: args.revision.to_ascii_lowercase(),
        path: args.filename.clone(),
        expected_size: args.expected_size,
        expected_sha256: expected_sha256.clone(),
        etag: remote.etag.clone(),
    };
    prepare_transfer_cache(&args.cache_dir, &identity)?;

    let partial_path = args
        .cache_dir
        .join(repo_folder)
        .join("blobs")
        .join(&remote.etag)
        .with_extension(HF_PARTIAL_EXTENSION);
    repair_invalid_partial(&partial_path, args.expected_size)?;

    let stop_hashing = Arc::new(AtomicBool::new(false));
    let hash_stop = Arc::clone(&stop_hashing);
    let hash_partial = partial_path.clone();
    let expected_size = args.expected_size;
    let hash_task = tokio::task::spawn_blocking(move || {
        hash_committed_prefix(&hash_partial, expected_size, &hash_stop)
    });

    let download_result = api_repo
        .download_with_progress(&args.filename, progress)
        .await;
    stop_hashing.store(true, Ordering::Release);
    let cursor = hash_task
        .await
        .context("joining incremental SHA-256 task")??;
    let pointer = download_result.context("hf-hub Range download failed")?;
    let blob = fs::canonicalize(&pointer)
        .with_context(|| format!("resolving hf-hub blob {}", pointer.display()))?;
    let actual_size = blob.metadata()?.len();
    if actual_size != args.expected_size {
        bail!(
            "hf-hub downloaded {} bytes for {}, expected {}",
            actual_size,
            args.filename,
            args.expected_size
        );
    }
    let actual_sha256 = finish_sha256(&blob, cursor, args.expected_size)?;
    if actual_sha256 != expected_sha256 {
        let _ = fs::remove_dir_all(&args.cache_dir);
        bail!(
            "downloaded file SHA-256 {} does not match expected {}",
            actual_sha256,
            expected_sha256
        );
    }

    if let Some(parent) = args.destination.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("creating staging directory {}", parent.display()))?;
    }
    fs::rename(&blob, &args.destination).with_context(|| {
        format!(
            "atomically moving verified blob {} to {}",
            blob.display(),
            args.destination.display()
        )
    })?;
    sync_file(&args.destination)?;
    if let Some(parent) = args.destination.parent() {
        sync_directory(parent)?;
    }
    let _ = fs::remove_dir_all(&args.cache_dir);
    let _ = fs::remove_file(&lock_path);
    emit_event(&TransferEvent::Complete {
        path: &args.destination,
        size: args.expected_size,
        sha256: &actual_sha256,
        etag: &remote.etag,
    });
    Ok(())
}

fn validate_args(args: &HfTransferArgs) -> Result<()> {
    let repo_parts = args.repo_id.split('/').collect::<Vec<_>>();
    if repo_parts.len() != 2 || repo_parts.iter().any(|part| part.is_empty()) {
        bail!("repo-id must use organization/model form");
    }
    if args.revision.len() != 40 || !args.revision.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("revision must be a full 40-character commit SHA");
    }
    if args.expected_sha256.len() != 64
        || !args
            .expected_sha256
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
    {
        bail!("expected-sha256 must contain 64 hexadecimal characters");
    }
    if args.parallelism == 0 {
        bail!("parallelism must be greater than zero");
    }
    if args.chunk_size == 0 {
        bail!("chunk-size must be greater than zero");
    }
    validate_repository_path(&args.filename)?;
    let cache_name = args
        .cache_dir
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    if !cache_name.ends_with(".hf-transfer") {
        bail!("cache-dir must end with .hf-transfer");
    }
    Ok(())
}

fn validate_repository_path(path: &str) -> Result<()> {
    let candidate = Path::new(path);
    if candidate.is_absolute() || path.is_empty() {
        bail!("filename must be a non-empty repository-relative path");
    }
    if candidate
        .components()
        .any(|component| !matches!(component, Component::Normal(_)))
    {
        bail!("filename contains an unsafe path component");
    }
    Ok(())
}

async fn resolve_remote_identity(url: &str, token: Option<&str>) -> Result<RemoteFileIdentity> {
    let mut headers = HeaderMap::new();
    headers.insert(USER_AGENT, HeaderValue::from_static("ironmlx/hf-transfer"));
    if let Some(token) = token {
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {token}"))
                .context("constructing Hugging Face authorization header")?,
        );
    }
    let client = reqwest::Client::builder()
        .default_headers(headers)
        .redirect(Policy::none())
        .build()
        .context("creating Hugging Face identity client")?;
    let response = client
        .get(url)
        .header(RANGE, "bytes=0-0")
        .send()
        .await
        .context("requesting immutable Hugging Face file identity")?;
    if !response.status().is_success() && !response.status().is_redirection() {
        bail!(
            "Hugging Face identity request failed with HTTP {}",
            response.status()
        );
    }
    let headers = response.headers();
    let commit_header = HeaderName::from_static("x-repo-commit");
    let linked_etag_header = HeaderName::from_static("x-linked-etag");
    let linked_size_header = HeaderName::from_static("x-linked-size");
    let commit_sha = required_header(headers, &commit_header)?;
    let etag = clean_etag(
        headers
            .get(&linked_etag_header)
            .or_else(|| headers.get(ETAG))
            .context("Hugging Face response is missing ETag")?
            .to_str()
            .context("Hugging Face returned a non-text ETag")?,
    )?;
    let size = if let Some(value) = headers.get(&linked_size_header) {
        value
            .to_str()
            .context("Hugging Face returned a non-text linked size")?
            .parse()
            .context("parsing Hugging Face linked size")?
    } else if let Some(value) = headers.get(CONTENT_RANGE) {
        parse_content_range_size(value.to_str()?)?
    } else if response.status().is_success() {
        required_header(headers, &CONTENT_LENGTH)?
            .parse()
            .context("parsing Hugging Face content length")?
    } else {
        let location = required_header(headers, &LOCATION)?;
        resolve_redirected_size(&location, token).await?
    };
    Ok(RemoteFileIdentity {
        commit_sha,
        size,
        etag,
    })
}

async fn resolve_redirected_size(location: &str, token: Option<&str>) -> Result<u64> {
    let client = reqwest::Client::new();
    let mut request = client.get(location).header(RANGE, "bytes=0-0");
    if let Some(token) = token {
        request = request.bearer_auth(token);
    }
    let response = request
        .send()
        .await
        .context("requesting redirected Hugging Face file size")?
        .error_for_status()
        .context("redirected Hugging Face file size request failed")?;
    let value = response
        .headers()
        .get(CONTENT_RANGE)
        .context("redirected Hugging Face response is missing Content-Range")?
        .to_str()?;
    parse_content_range_size(value)
}

fn required_header(headers: &HeaderMap, name: &HeaderName) -> Result<String> {
    headers
        .get(name)
        .with_context(|| format!("Hugging Face response is missing {name}"))?
        .to_str()
        .with_context(|| format!("Hugging Face returned a non-text {name}"))
        .map(str::to_string)
}

fn parse_content_range_size(value: &str) -> Result<u64> {
    value
        .rsplit_once('/')
        .context("invalid Hugging Face Content-Range")?
        .1
        .parse()
        .context("parsing Hugging Face Content-Range size")
}

fn clean_etag(value: &str) -> Result<String> {
    let etag = value.trim().trim_matches('"').to_string();
    if etag.is_empty() || etag == "." || etag == ".." || etag.contains('/') || etag.contains('\\') {
        bail!("Hugging Face returned an unsafe ETag");
    }
    Ok(etag)
}

fn prepare_transfer_cache(cache_dir: &Path, identity: &TransferIdentity) -> Result<()> {
    let identity_path = cache_dir.join("identity.json");
    let matches = fs::read(&identity_path)
        .ok()
        .and_then(|data| serde_json::from_slice::<TransferIdentity>(&data).ok())
        .is_some_and(|stored| stored == *identity);
    if cache_dir.exists() && !matches {
        fs::remove_dir_all(cache_dir).with_context(|| {
            format!(
                "discarding partial with mismatched file identity {}",
                cache_dir.display()
            )
        })?;
    }
    fs::create_dir_all(cache_dir)
        .with_context(|| format!("creating transfer cache {}", cache_dir.display()))?;
    atomic_write_json(&identity_path, identity)
}

fn atomic_write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let parent = path.parent().context("identity path has no parent")?;
    fs::create_dir_all(parent)?;
    let temporary = parent.join(format!(".identity.{}.tmp", uuid::Uuid::new_v4()));
    let data = serde_json::to_vec_pretty(value)?;
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temporary)?;
    file.write_all(&data)?;
    file.sync_all()?;
    fs::rename(&temporary, path)?;
    sync_directory(parent)
}

fn repair_invalid_partial(path: &Path, expected_size: u64) -> Result<()> {
    if !path.exists() {
        return Ok(());
    }
    let mut file = File::open(path)?;
    let expected_storage_size = expected_size
        .checked_add(std::mem::size_of::<u64>() as u64)
        .context("partial size overflow")?;
    if file.metadata()?.len() != expected_storage_size {
        drop(file);
        fs::remove_file(path)?;
        return Ok(());
    }
    file.seek(SeekFrom::Start(expected_size))?;
    let mut marker = [0_u8; std::mem::size_of::<u64>()];
    file.read_exact(&mut marker)?;
    let committed = u64::from_le_bytes(marker);
    if committed == expected_size.saturating_add(1) {
        drop(file);
        let mut file = OpenOptions::new().write(true).open(path)?;
        file.seek(SeekFrom::Start(expected_size))?;
        file.write_all(&expected_size.to_le_bytes())?;
        file.sync_all()?;
    } else if committed > expected_size {
        drop(file);
        fs::remove_file(path)?;
    }
    Ok(())
}

fn hash_committed_prefix(
    partial: &Path,
    expected_size: u64,
    stop: &AtomicBool,
) -> Result<HashCursor> {
    let mut cursor = HashCursor {
        digest: Sha256::new(),
        bytes_hashed: 0,
    };
    loop {
        if let Some(committed) = committed_prefix(partial, expected_size)? {
            if committed > cursor.bytes_hashed {
                hash_file_range(partial, &mut cursor.digest, cursor.bytes_hashed, committed)?;
                cursor.bytes_hashed = committed;
            }
        }
        if stop.load(Ordering::Acquire) {
            return Ok(cursor);
        }
        std::thread::sleep(HASH_POLL_INTERVAL);
    }
}

fn committed_prefix(partial: &Path, expected_size: u64) -> Result<Option<u64>> {
    let Ok(mut file) = File::open(partial) else {
        return Ok(None);
    };
    let marker_end = expected_size
        .checked_add(std::mem::size_of::<u64>() as u64)
        .context("partial marker offset overflow")?;
    if file.metadata()?.len() != marker_end {
        return Ok(None);
    }
    file.seek(SeekFrom::Start(expected_size))?;
    let mut marker = [0_u8; std::mem::size_of::<u64>()];
    file.read_exact(&mut marker)?;
    let committed = u64::from_le_bytes(marker);
    if committed == expected_size.saturating_add(1) {
        return Ok(Some(expected_size));
    }
    if committed > expected_size {
        bail!("hf-hub partial contains an invalid committed byte marker");
    }
    Ok(Some(committed))
}

fn finish_sha256(path: &Path, mut cursor: HashCursor, expected_size: u64) -> Result<String> {
    if cursor.bytes_hashed < expected_size {
        hash_file_range(path, &mut cursor.digest, cursor.bytes_hashed, expected_size)?;
        cursor.bytes_hashed = expected_size;
    }
    if cursor.bytes_hashed != expected_size {
        bail!("incremental SHA-256 did not cover the complete file");
    }
    Ok(format!("{:x}", cursor.digest.finalize()))
}

fn hash_file_range(path: &Path, digest: &mut Sha256, start: u64, end: u64) -> Result<()> {
    if start > end {
        bail!("invalid SHA-256 file range");
    }
    let mut file = File::open(path)
        .with_context(|| format!("opening {} for incremental SHA-256", path.display()))?;
    file.seek(SeekFrom::Start(start))?;
    let mut remaining = end - start;
    let mut buffer = vec![0_u8; 4 * 1024 * 1024];
    while remaining > 0 {
        let requested = usize::try_from(remaining.min(buffer.len() as u64))?;
        let read = file.read(&mut buffer[..requested])?;
        if read == 0 {
            bail!(
                "unexpected EOF hashing {} at byte {}",
                path.display(),
                end - remaining
            );
        }
        digest.update(&buffer[..read]);
        remaining -= read as u64;
    }
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String> {
    let size = path.metadata()?.len();
    finish_sha256(
        path,
        HashCursor {
            digest: Sha256::new(),
            bytes_hashed: 0,
        },
        size,
    )
}

fn transfer_lock_path(cache_dir: &Path) -> Result<PathBuf> {
    let name = cache_dir
        .file_name()
        .and_then(|name| name.to_str())
        .context("cache-dir has no valid final component")?;
    Ok(cache_dir.with_file_name(format!("{name}.lock")))
}

fn sync_file(path: &Path) -> Result<()> {
    OpenOptions::new()
        .read(true)
        .open(path)?
        .sync_all()
        .with_context(|| format!("syncing verified file {}", path.display()))
}

fn sync_directory(path: &Path) -> Result<()> {
    File::open(path)?
        .sync_all()
        .with_context(|| format!("syncing directory {}", path.display()))
}

fn emit_event(event: &TransferEvent<'_>) {
    if let Ok(json) = serde_json::to_string(event) {
        println!("{json}");
        let _ = std::io::stdout().flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::extract::State;
    use axum::http::{Request, Response, StatusCode};
    use axum::routing::get;
    use axum::Router;

    #[derive(Clone)]
    struct TestHubState {
        payload: Arc<Vec<u8>>,
        commit: String,
        etag: String,
        ranges: Arc<Mutex<Vec<String>>>,
    }

    fn args(cache_dir: PathBuf) -> HfTransferArgs {
        HfTransferArgs {
            repo_id: "org/model".to_string(),
            revision: "a".repeat(40),
            filename: "weights/model.safetensors".to_string(),
            destination: cache_dir
                .parent()
                .expect("cache parent")
                .join("model.safetensors"),
            cache_dir,
            expected_size: 16,
            expected_sha256: "b".repeat(64),
            endpoint: "https://huggingface.co".to_string(),
            parallelism: 4,
            chunk_size: 10_000_000,
        }
    }

    #[test]
    fn validates_immutable_identity_and_scoped_cache() {
        let valid = args(PathBuf::from("/tmp/model.safetensors.hf-transfer"));
        validate_args(&valid).expect("valid arguments");

        let mut mutable = valid;
        mutable.revision = "main".to_string();
        assert!(validate_args(&mutable).is_err());
    }

    #[test]
    fn changed_identity_discards_only_transfer_cache() {
        let root =
            std::env::temp_dir().join(format!("ironmlx-hf-transfer-{}", uuid::Uuid::new_v4()));
        let cache = root.join("model.safetensors.hf-transfer");
        fs::create_dir_all(&cache).expect("create cache");
        fs::write(cache.join("stale.sync.part"), b"stale").expect("write stale partial");
        let first = TransferIdentity {
            version: TRANSFER_IDENTITY_VERSION,
            provider: "huggingface".to_string(),
            repo_id: "org/model".to_string(),
            commit_sha: "a".repeat(40),
            path: "model.safetensors".to_string(),
            expected_size: 5,
            expected_sha256: "b".repeat(64),
            etag: "old".to_string(),
        };
        atomic_write_json(&cache.join("identity.json"), &first).expect("write first identity");
        let second = TransferIdentity {
            etag: "new".to_string(),
            ..first
        };

        prepare_transfer_cache(&cache, &second).expect("prepare changed identity");

        assert!(!cache.join("stale.sync.part").exists());
        let stored: TransferIdentity =
            serde_json::from_slice(&fs::read(cache.join("identity.json")).expect("read identity"))
                .expect("decode identity");
        assert_eq!(stored, second);
        fs::remove_dir_all(root).expect("remove test root");
    }

    #[test]
    fn hashes_only_committed_prefix_then_finishes_without_rehashing() {
        let root = std::env::temp_dir().join(format!("ironmlx-hf-hash-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).expect("create root");
        let partial = root.join("blob.sync.part");
        let payload = b"abcdefghijklmnop";
        let mut file = File::create(&partial).expect("create partial");
        file.set_len((payload.len() + std::mem::size_of::<u64>()) as u64)
            .expect("preallocate partial");
        file.seek(SeekFrom::Start(0)).expect("seek payload");
        file.write_all(payload).expect("write payload");
        file.seek(SeekFrom::Start(payload.len() as u64))
            .expect("seek marker");
        file.write_all(&8_u64.to_le_bytes())
            .expect("write committed marker");
        file.sync_all().expect("sync partial");

        let stop = AtomicBool::new(true);
        let cursor =
            hash_committed_prefix(&partial, payload.len() as u64, &stop).expect("hash prefix");
        assert_eq!(cursor.bytes_hashed, 8);
        file.set_len(payload.len() as u64).expect("truncate marker");
        let actual = finish_sha256(&partial, cursor, payload.len() as u64).expect("finish SHA-256");
        let expected = format!("{:x}", Sha256::digest(payload));
        assert_eq!(actual, expected);
        fs::remove_dir_all(root).expect("remove test root");
    }

    #[test]
    fn invalid_partial_marker_is_removed_before_hf_hub_resume() {
        let root = std::env::temp_dir().join(format!("ironmlx-hf-marker-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).expect("create root");
        let partial = root.join("blob.sync.part");
        let mut file = File::create(&partial).expect("create partial");
        file.set_len(24).expect("preallocate partial");
        file.seek(SeekFrom::Start(16)).expect("seek marker");
        file.write_all(&18_u64.to_le_bytes())
            .expect("write invalid marker");
        file.sync_all().expect("sync partial");

        repair_invalid_partial(&partial, 16).expect("repair partial");

        assert!(!partial.exists());
        fs::remove_dir_all(root).expect("remove test root");
    }

    #[tokio::test]
    async fn json_progress_uses_first_update_as_absolute_resume_baseline() {
        let mut progress = JsonProgress::default();

        progress.init(100, "model.safetensors").await;
        {
            let state = progress.state.lock().expect("progress state");
            assert!(state.awaiting_baseline);
            assert_eq!(state.bytes, 0);
            assert_eq!(state.last_emitted_bytes, 0);
        }

        progress.update(20).await;
        {
            let state = progress.state.lock().expect("progress state");
            assert!(!state.awaiting_baseline);
            assert_eq!(state.bytes, 20);
            assert_eq!(state.last_emitted_bytes, 20);
            assert_eq!(state.first_emitted_bytes, Some(20));
        }

        progress.update(7).await;
        let state = progress.state.lock().expect("progress state");
        assert_eq!(state.bytes, 27);
        assert_eq!(state.last_emitted_bytes, 20);
    }

    #[tokio::test]
    async fn json_progress_reports_zero_when_hf_hub_discards_partial() {
        let mut progress = JsonProgress::default();

        progress.init(100, "model.safetensors").await;
        progress.update(0).await;

        let state = progress.state.lock().expect("progress state");
        assert!(!state.awaiting_baseline);
        assert_eq!(state.bytes, 0);
        assert_eq!(state.last_emitted_bytes, 0);
        assert_eq!(state.first_emitted_bytes, Some(0));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn hf_hub_resumes_committed_ranges_and_publishes_verified_destination() {
        let payload = Arc::new(
            (0..(2 * 1024 * 1024 + 137))
                .map(|index| (index % 251) as u8)
                .collect::<Vec<_>>(),
        );
        let expected_sha256 = format!("{:x}", Sha256::digest(payload.as_slice()));
        let commit = "c".repeat(40);
        let etag = expected_sha256.clone();
        let ranges = Arc::new(Mutex::new(Vec::new()));
        let state = TestHubState {
            payload: Arc::clone(&payload),
            commit: commit.clone(),
            etag: etag.clone(),
            ranges: Arc::clone(&ranges),
        };
        let app = Router::new()
            .route("/*path", get(test_hub_file))
            .with_state(state);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind test hub");
        let address = listener.local_addr().expect("test hub address");
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.expect("serve test hub");
        });

        let root = std::env::temp_dir().join(format!("ironmlx-hf-e2e-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).expect("create root");
        let cache_dir = root.join("model.safetensors.hf-transfer");
        let destination = root.join("model.safetensors");
        let identity = TransferIdentity {
            version: TRANSFER_IDENTITY_VERSION,
            provider: "huggingface".to_string(),
            repo_id: "org/model".to_string(),
            commit_sha: commit.clone(),
            path: "model.safetensors".to_string(),
            expected_size: payload.len() as u64,
            expected_sha256: expected_sha256.clone(),
            etag: etag.clone(),
        };
        prepare_transfer_cache(&cache_dir, &identity).expect("prepare transfer cache");
        let partial = cache_dir
            .join("models--org--model")
            .join("blobs")
            .join(&etag)
            .with_extension(HF_PARTIAL_EXTENSION);
        fs::create_dir_all(partial.parent().expect("partial parent")).expect("create blob dir");
        let resume_offset = 512 * 1024_u64;
        let mut file = File::create(&partial).expect("create partial");
        file.set_len(payload.len() as u64 + std::mem::size_of::<u64>() as u64)
            .expect("preallocate partial");
        file.seek(SeekFrom::Start(0)).expect("seek partial");
        file.write_all(&payload[..resume_offset as usize])
            .expect("write committed prefix");
        file.seek(SeekFrom::Start(payload.len() as u64))
            .expect("seek committed marker");
        file.write_all(&resume_offset.to_le_bytes())
            .expect("write committed marker");
        file.sync_all().expect("sync partial");

        let progress = JsonProgress::default();
        run_async_with_progress(
            HfTransferArgs {
                repo_id: "org/model".to_string(),
                revision: commit,
                filename: "model.safetensors".to_string(),
                destination: destination.clone(),
                cache_dir: cache_dir.clone(),
                expected_size: payload.len() as u64,
                expected_sha256: expected_sha256.clone(),
                endpoint: format!("http://{address}"),
                parallelism: 4,
                chunk_size: 256 * 1024,
            },
            progress.clone(),
        )
        .await
        .expect("run hf-hub transfer");

        assert_eq!(
            progress
                .state
                .lock()
                .expect("progress state")
                .first_emitted_bytes,
            Some(resume_offset)
        );

        assert_eq!(fs::read(&destination).expect("read destination"), *payload);
        assert_eq!(
            sha256_file(&destination).expect("hash destination"),
            expected_sha256
        );
        assert!(!cache_dir.exists());
        assert!(!transfer_lock_path(&cache_dir).expect("lock path").exists());
        let requested = ranges.lock().expect("ranges lock").clone();
        assert!(
            requested
                .iter()
                .filter(|range| range.as_str() != "bytes=0-0")
                .all(|range| range_start(range) >= resume_offset),
            "unexpected pre-resume Range request: {requested:?}"
        );

        server.abort();
        fs::remove_dir_all(root).expect("remove test root");
    }

    async fn test_hub_file(
        State(state): State<TestHubState>,
        request: Request<Body>,
    ) -> Response<Body> {
        let range = request
            .headers()
            .get(RANGE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or("bytes=0-0")
            .to_string();
        state
            .ranges
            .lock()
            .expect("ranges lock")
            .push(range.clone());
        let start = range_start(&range) as usize;
        let requested_end = range
            .strip_prefix("bytes=")
            .and_then(|value| value.split_once('-'))
            .and_then(|(_, end)| end.parse::<usize>().ok())
            .unwrap_or(start);
        let end = requested_end.min(state.payload.len() - 1);
        let body = Body::from(state.payload[start..=end].to_vec());
        Response::builder()
            .status(StatusCode::PARTIAL_CONTENT)
            .header("x-repo-commit", &state.commit)
            .header("x-linked-etag", format!("\"{}\"", state.etag))
            .header(
                CONTENT_RANGE,
                format!("bytes {start}-{end}/{}", state.payload.len()),
            )
            .header(CONTENT_LENGTH, end - start + 1)
            .body(body)
            .expect("test response")
    }

    fn range_start(value: &str) -> u64 {
        value
            .strip_prefix("bytes=")
            .and_then(|value| value.split_once('-'))
            .and_then(|(start, _)| start.parse().ok())
            .expect("valid byte range")
    }
}
