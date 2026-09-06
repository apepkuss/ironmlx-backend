use std::path::PathBuf;
use std::process::Command;

/// Build an `ironmlx` command pinned to the metallib installed with `MLX_DIR`.
///
/// Real-model subprocess tests must not depend on MLX's compiled-in metallib
/// path or on the child's working directory. The global CLI option must appear
/// before the subcommand appended by the caller.
pub fn command() -> Command {
    let mlx_dir = std::env::var_os("MLX_DIR").expect("MLX_DIR must be set");
    let metallib = PathBuf::from(&mlx_dir).join("lib/mlx.metallib");
    let metallib = metallib.canonicalize().unwrap_or_else(|error| {
        panic!(
            "MLX_DIR/lib/mlx.metallib must resolve to a regular file at {}: {error}",
            metallib.display()
        )
    });
    assert!(
        metallib.is_file(),
        "MLX_DIR/lib/mlx.metallib must be a regular file: {}",
        metallib.display()
    );

    let mut command = Command::new(env!("CARGO_BIN_EXE_ironmlx"));
    command
        .arg("--mlx-metallib")
        .arg(metallib)
        .env("MLX_DIR", mlx_dir);
    command
}
