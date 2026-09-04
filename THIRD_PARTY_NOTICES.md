# IronMLX Third-Party Notices

This engineering inventory describes third-party software and assets included in
the Apple Silicon macOS Release product. It is generated from the locked Rust
dependency graph, pinned native and Swift inputs, and reviewed bundled
third-party assets. It is not legal advice or approval to distribute the
product.

## Native dependencies

| Component | Revision | License | License text | Source |
|---|---|---|---|---|
| fmt | `12.1.0` | MIT | `THIRD_PARTY_LICENSES/native-fmt-mit.txt` | https://github.com/fmtlib/fmt.git |
| gguflib | `8fa6eb65236618e28fd7710a0fba565f7faa1848` | MIT | `THIRD_PARTY_LICENSES/native-gguflib-mit.txt` | https://github.com/antirez/gguf-tools.git |
| metal-cpp | `metal-cpp_26.zip` | Apache-2.0 | `THIRD_PARTY_LICENSES/native-metal-cpp-apache-2.0.txt` | https://developer.apple.com/metal/cpp/ |
| MLX C++ with bundled JACCL | `73ad5df20cb30be4192e5c4d0ae8130674773427` | MIT | `THIRD_PARTY_LICENSES/native-mlx-mit.txt` | https://github.com/apepkuss/mlx.git |
| nlohmann/json | `3.11.3` | MIT | `THIRD_PARTY_LICENSES/native-nlohmann-json-mit.txt` | https://github.com/nlohmann/json.git |

The MLX entry identifies the non-official IronMLX fork and its exact
revision. Bundled JACCL sources are part of that checkout and are covered
by the checkout's MLX license file.

## Rust dependencies

Scope: `ironmlx` and `iron-bench` Release binaries for
`aarch64-apple-darwin`, default features, locked dependency graphs,
development dependencies excluded, build dependencies retained.

| Crate | Version | License expression | License text | Source |
|---|---|---|---|---|
| adler2 | 2.0.1 | 0BSD OR MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-23f18e03dc49df91.txt` | https://github.com/oyvindln/adler2 |
| ahash | 0.8.12 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-0444c6991eead682.txt` | https://github.com/tkaitchuck/ahash |
| aho-corasick | 1.1.4 | Unlicense OR MIT | `THIRD_PARTY_LICENSES/rust-license-0f96a83840e146e4.txt` | https://github.com/BurntSushi/aho-corasick |
| anstream | 1.0.0 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-6efb0476a1cc0850.txt` | https://github.com/rust-cli/anstyle.git |
| anstyle | 1.0.14 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-6efb0476a1cc0850.txt` | https://github.com/rust-cli/anstyle.git |
| anstyle-parse | 1.0.0 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-6efb0476a1cc0850.txt` | https://github.com/rust-cli/anstyle.git |
| anstyle-query | 1.1.5 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-6efb0476a1cc0850.txt` | https://github.com/rust-cli/anstyle.git |
| anyhow | 1.0.104 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-23f18e03dc49df91.txt` | https://github.com/dtolnay/anyhow |
| arc-swap | 1.9.2 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-ff3f1cd12af8866d.txt` | https://github.com/vorner/arc-swap |
| async-channel | 2.5.0 | Apache-2.0 OR MIT | `THIRD_PARTY_LICENSES/rust-license-23f18e03dc49df91.txt` | https://github.com/smol-rs/async-channel |
| async-task | 4.7.1 | Apache-2.0 OR MIT | `THIRD_PARTY_LICENSES/rust-license-23f18e03dc49df91.txt` | https://github.com/smol-rs/async-task |
| async-trait | 0.1.91 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-23f18e03dc49df91.txt` | https://github.com/dtolnay/async-trait |
| atomic-waker | 1.1.2 | Apache-2.0 OR MIT | `THIRD_PARTY_LICENSES/rust-license-23f18e03dc49df91.txt` | https://github.com/smol-rs/atomic-waker |
| autocfg | 1.5.1 | Apache-2.0 OR MIT | `THIRD_PARTY_LICENSES/rust-license-27995d58ad5c1145.txt` | https://github.com/cuviper/autocfg |
| axum | 0.7.9 | MIT | `THIRD_PARTY_LICENSES/rust-license-c14b6ed9d732322a.txt` | https://github.com/tokio-rs/axum |
| axum-core | 0.4.5 | MIT | `THIRD_PARTY_LICENSES/rust-license-ab25eee08e7b6d20.txt` | https://github.com/tokio-rs/axum |
| axum-macros | 0.4.2 | MIT | `THIRD_PARTY_LICENSES/rust-license-ab25eee08e7b6d20.txt` | https://github.com/tokio-rs/axum |
| axum-server | 0.7.3 | MIT | `THIRD_PARTY_LICENSES/rust-license-ab743bd126625ce5.txt` | https://github.com/programatik29/axum-server |
| base64 | 0.13.1 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-0dd882e53de11566.txt` | https://github.com/marshallpierce/rust-base64 |
| base64 | 0.22.1 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-0dd882e53de11566.txt` | https://github.com/marshallpierce/rust-base64 |
| bitflags | 2.13.1 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-6485b8ed310d3f03.txt` | https://github.com/bitflags/bitflags |
| block-buffer | 0.10.4 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-d5c22aa3118d240e.txt` | https://github.com/RustCrypto/utils |
| blocking | 1.6.2 | Apache-2.0 OR MIT | `THIRD_PARTY_LICENSES/rust-license-23f18e03dc49df91.txt` | https://github.com/smol-rs/blocking |
| borrow-or-share | 0.2.4 | MIT-0 | `THIRD_PARTY_LICENSES/rust-license-fdef904ef5d29e4d.txt` | https://github.com/yescallop/borrow-or-share |
| bytemuck | 1.25.2 | Zlib OR Apache-2.0 OR MIT | `THIRD_PARTY_LICENSES/rust-license-9df9ba60a11af705.txt` | https://github.com/Lokathor/bytemuck |
| bytemuck_derive | 1.11.0 | Zlib OR Apache-2.0 OR MIT | `THIRD_PARTY_LICENSES/rust-license-9df9ba60a11af705.txt` | https://github.com/Lokathor/bytemuck |
| byteorder-lite | 0.1.0 | Unlicense OR MIT | `THIRD_PARTY_LICENSES/rust-license-0f96a83840e146e4.txt` | https://github.com/image-rs/byteorder-lite |
| bytes | 1.12.1 | MIT | `THIRD_PARTY_LICENSES/rust-license-45f522cacecb1023.txt` | https://github.com/tokio-rs/bytes |
| cc | 1.4.0 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-378f5840b258e277.txt` | https://github.com/rust-lang/cc-rs |
| cfg-if | 1.0.4 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-378f5840b258e277.txt` | https://github.com/rust-lang/cfg-if |
| clap | 4.6.5 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-6efb0476a1cc0850.txt` | https://github.com/clap-rs/clap |
| clap_builder | 4.6.5 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-6efb0476a1cc0850.txt` | https://github.com/clap-rs/clap |
| clap_derive | 4.6.4 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-6efb0476a1cc0850.txt` | https://github.com/clap-rs/clap |
| clap_lex | 1.1.0 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-6efb0476a1cc0850.txt` | https://github.com/clap-rs/clap |
| codespan-reporting | 0.13.1 | Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-50e6751797c50ded.txt` | https://github.com/brendanzab/codespan |
| colorchoice | 1.0.5 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-6efb0476a1cc0850.txt` | https://github.com/rust-cli/anstyle.git |
| concurrent-queue | 2.5.0 | Apache-2.0 OR MIT | `THIRD_PARTY_LICENSES/rust-license-23f18e03dc49df91.txt` | https://github.com/smol-rs/concurrent-queue |
| console | 0.15.11 | MIT | `THIRD_PARTY_LICENSES/rust-license-3e1de3c527ab2512.txt` | https://github.com/console-rs/console |
| core-foundation | 0.9.4 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-62065228e42caebc.txt` | https://github.com/servo/core-foundation-rs |
| core-foundation-sys | 0.8.7 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-62065228e42caebc.txt` | https://github.com/servo/core-foundation-rs |
| cpufeatures | 0.2.17 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-ae9baa7beea91027.txt` | https://github.com/RustCrypto/utils |
| crc32fast | 1.5.0 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-61d383b05b87d78f.txt` | https://github.com/srijs/rust-crc32fast |
| crossbeam-deque | 0.8.7 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-5734ed989dfca1f6.txt` | https://github.com/crossbeam-rs/crossbeam |
| crossbeam-epoch | 0.9.20 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-5734ed989dfca1f6.txt` | https://github.com/crossbeam-rs/crossbeam |
| crossbeam-utils | 0.8.22 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-5734ed989dfca1f6.txt` | https://github.com/crossbeam-rs/crossbeam |
| crypto-common | 0.1.7 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-3521672491a34794.txt` | https://github.com/RustCrypto/traits |
| cxx | 1.0.198 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-23f18e03dc49df91.txt` | https://github.com/dtolnay/cxx |
| cxx-build | 1.0.198 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-23f18e03dc49df91.txt` | https://github.com/dtolnay/cxx |
| cxxbridge-flags | 1.0.198 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-23f18e03dc49df91.txt` | https://github.com/dtolnay/cxx |
| cxxbridge-macro | 1.0.198 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-23f18e03dc49df91.txt` | https://github.com/dtolnay/cxx |
| darling | 0.20.11 | MIT | `THIRD_PARTY_LICENSES/rust-license-8ea93490d74a5a1b.txt` | https://github.com/TedDriggs/darling |
| darling_core | 0.20.11 | MIT | `THIRD_PARTY_LICENSES/rust-license-8ea93490d74a5a1b.txt` | https://github.com/TedDriggs/darling |
| darling_macro | 0.20.11 | MIT | `THIRD_PARTY_LICENSES/rust-license-8ea93490d74a5a1b.txt` | https://github.com/TedDriggs/darling |
| derive_builder | 0.20.2 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-8c9612877aacfa1b.txt` | https://github.com/colin-kiegel/rust-derive-builder |
| derive_builder_core | 0.20.2 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-8c9612877aacfa1b.txt` | https://github.com/colin-kiegel/rust-derive-builder |
| derive_builder_macro | 0.20.2 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-8c9612877aacfa1b.txt` | https://github.com/colin-kiegel/rust-derive-builder |
| derivre | 0.3.12 | MIT | `THIRD_PARTY_LICENSES/rust-license-3d4ada4e04d153d7.txt` | https://github.com/microsoft/derivre |
| digest | 0.10.7 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-9e0dfd2dd4173a53.txt` | https://github.com/RustCrypto/traits |
| dirs | 6.0.0 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-6a2e0ade09a7d5f8.txt` | https://github.com/soc/dirs-rs |
| dirs-sys | 0.5.0 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-6a2e0ade09a7d5f8.txt` | https://github.com/dirs-dev/dirs-sys-rs |
| displaydoc | 0.2.7 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-23f18e03dc49df91.txt` | https://github.com/yaahc/displaydoc |
| either | 1.17.0 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-7576269ea71f767b.txt` | https://github.com/rayon-rs/either |
| encoding_rs | 0.8.35 | (Apache-2.0 OR MIT) AND BSD-3-Clause | `THIRD_PARTY_LICENSES/rust-license-3fa4ca83dcc92378.txt`<br>`THIRD_PARTY_LICENSES/rust-license-838118388fe5c2e7.txt` | https://github.com/hsivonen/encoding_rs |
| equivalent | 1.0.2 | Apache-2.0 OR MIT | `THIRD_PARTY_LICENSES/rust-license-7365cc8878a1d7ce.txt` | https://github.com/indexmap-rs/equivalent |
| esaxx-rs | 0.1.10 | Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-50e6751797c50ded.txt` | https://github.com/Narsil/esaxx-rs |
| event-listener | 5.4.2 | Apache-2.0 OR MIT | `THIRD_PARTY_LICENSES/rust-license-23f18e03dc49df91.txt` | https://github.com/smol-rs/event-listener |
| event-listener-strategy | 0.5.4 | Apache-2.0 OR MIT | `THIRD_PARTY_LICENSES/rust-license-23f18e03dc49df91.txt` | https://github.com/smol-rs/event-listener-strategy |
| fastrand | 2.5.0 | Apache-2.0 OR MIT | `THIRD_PARTY_LICENSES/rust-license-23f18e03dc49df91.txt` | https://github.com/smol-rs/fastrand |
| fdeflate | 0.3.7 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-c77a4cf9da729987.txt` | https://github.com/image-rs/fdeflate |
| find-msvc-tools | 0.1.9 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-378f5840b258e277.txt` | https://github.com/rust-lang/cc-rs |
| flate2 | 1.1.9 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-025436edff4cfcdd.txt` | https://github.com/rust-lang/flate2-rs |
| fluent-uri | 0.3.2 | MIT | `THIRD_PARTY_LICENSES/rust-license-e699bec719875d8e.txt` | https://github.com/yescallop/fluent-uri-rs |
| fnv | 1.0.7 | Apache-2.0  OR  MIT | `THIRD_PARTY_LICENSES/rust-license-65fdb6c76cd61612.txt` | https://github.com/servo/rust-fnv |
| foldhash | 0.2.0 | Zlib | `THIRD_PARTY_LICENSES/rust-license-1d4c38d56650edc2.txt` | https://github.com/orlp/foldhash |
| form_urlencoded | 1.2.2 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-20c7855c364d57ea.txt` | https://github.com/servo/rust-url |
| fs-err | 3.3.1 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-ebeeae2d65a7fc03.txt` | https://github.com/andrewhickman/fs-err |
| futures | 0.3.33 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-6652c868f35dfe5e.txt` | https://github.com/rust-lang/futures-rs |
| futures-channel | 0.3.33 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-6652c868f35dfe5e.txt` | https://github.com/rust-lang/futures-rs |
| futures-core | 0.3.33 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-6652c868f35dfe5e.txt` | https://github.com/rust-lang/futures-rs |
| futures-executor | 0.3.33 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-6652c868f35dfe5e.txt` | https://github.com/rust-lang/futures-rs |
| futures-io | 0.3.33 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-6652c868f35dfe5e.txt` | https://github.com/rust-lang/futures-rs |
| futures-lite | 2.6.1 | Apache-2.0 OR MIT | `THIRD_PARTY_LICENSES/rust-license-23f18e03dc49df91.txt` | https://github.com/smol-rs/futures-lite |
| futures-macro | 0.3.33 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-6652c868f35dfe5e.txt` | https://github.com/rust-lang/futures-rs |
| futures-sink | 0.3.33 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-6652c868f35dfe5e.txt` | https://github.com/rust-lang/futures-rs |
| futures-task | 0.3.33 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-6652c868f35dfe5e.txt` | https://github.com/rust-lang/futures-rs |
| futures-util | 0.3.33 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-6652c868f35dfe5e.txt` | https://github.com/rust-lang/futures-rs |
| generic-array | 0.14.7 | MIT | `THIRD_PARTY_LICENSES/rust-license-8a28736d1243c67e.txt` | https://github.com/fizyk20/generic-array.git |
| getrandom | 0.2.17 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-42fa16951ce7f24b.txt` | https://github.com/rust-random/getrandom |
| getrandom | 0.3.4 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-29e9fe5074bd27e0.txt` | https://github.com/rust-random/getrandom |
| getrandom | 0.4.3 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-523a42c25d245dde.txt` | https://github.com/rust-random/getrandom |
| h2 | 0.4.16 | MIT | `THIRD_PARTY_LICENSES/rust-license-b21623012e6c453d.txt` | https://github.com/hyperium/h2 |
| half | 2.7.1 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-508a77d2e7b51d98.txt` | https://github.com/VoidStarKat/half-rs |
| hashbrown | 0.17.1 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-ff8f68cb076caf8c.txt` | https://github.com/rust-lang/hashbrown |
| heck | 0.5.0 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-7b63ecd5f1902af1.txt` | https://github.com/withoutboats/heck |
| hf-hub | 0.4.3 | Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-50e6751797c50ded.txt` | https://github.com/huggingface/hf-hub |
| http | 1.5.0 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-dc91f8200e4b2a1f.txt` | https://github.com/hyperium/http |
| http-body | 1.1.0 | MIT | `THIRD_PARTY_LICENSES/rust-license-248378d0a3383c17.txt` | https://github.com/hyperium/http-body |
| http-body-util | 0.1.4 | MIT | `THIRD_PARTY_LICENSES/rust-license-248378d0a3383c17.txt` | https://github.com/hyperium/http-body |
| httparse | 1.10.1 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-1626f2c950cee975.txt` | https://github.com/seanmonstar/httparse |
| httpdate | 1.0.3 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-934887691e05d69d.txt` | https://github.com/pyfisch/httpdate |
| hyper | 1.11.0 | MIT | `THIRD_PARTY_LICENSES/rust-license-2d01890414494742.txt` | https://github.com/hyperium/hyper |
| hyper-rustls | 0.27.9 | Apache-2.0 OR ISC OR MIT | `THIRD_PARTY_LICENSES/rust-license-709e3175b4212f7b.txt` | https://github.com/rustls/hyper-rustls |
| hyper-util | 0.1.20 | MIT | `THIRD_PARTY_LICENSES/rust-license-9e0a97848ea543ae.txt` | https://github.com/hyperium/hyper-util |
| icu_collections | 2.2.0 | Unicode-3.0 | `THIRD_PARTY_LICENSES/rust-license-f367c1b8e1aa2624.txt` | https://github.com/unicode-org/icu4x |
| icu_locale_core | 2.2.0 | Unicode-3.0 | `THIRD_PARTY_LICENSES/rust-license-f367c1b8e1aa2624.txt` | https://github.com/unicode-org/icu4x |
| icu_normalizer | 2.2.0 | Unicode-3.0 | `THIRD_PARTY_LICENSES/rust-license-f367c1b8e1aa2624.txt` | https://github.com/unicode-org/icu4x |
| icu_normalizer_data | 2.2.0 | Unicode-3.0 | `THIRD_PARTY_LICENSES/rust-license-f367c1b8e1aa2624.txt` | https://github.com/unicode-org/icu4x |
| icu_properties | 2.2.0 | Unicode-3.0 | `THIRD_PARTY_LICENSES/rust-license-f367c1b8e1aa2624.txt` | https://github.com/unicode-org/icu4x |
| icu_properties_data | 2.2.0 | Unicode-3.0 | `THIRD_PARTY_LICENSES/rust-license-f367c1b8e1aa2624.txt` | https://github.com/unicode-org/icu4x |
| icu_provider | 2.2.0 | Unicode-3.0 | `THIRD_PARTY_LICENSES/rust-license-f367c1b8e1aa2624.txt` | https://github.com/unicode-org/icu4x |
| ident_case | 1.0.1 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-508a77d2e7b51d98.txt` | https://github.com/TedDriggs/ident_case |
| idna | 1.1.0 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-b38f11f6096706e6.txt` | https://github.com/servo/rust-url/ |
| idna_adapter | 1.2.2 | Apache-2.0 OR MIT | `THIRD_PARTY_LICENSES/rust-license-8b43ce8accd61e9d.txt` | https://github.com/hsivonen/idna_adapter |
| image | 0.25.10 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-c77a4cf9da729987.txt` | https://github.com/image-rs/image |
| image-webp | 0.2.4 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-c77a4cf9da729987.txt` | https://github.com/image-rs/image-webp |
| indexmap | 2.14.0 | Apache-2.0 OR MIT | `THIRD_PARTY_LICENSES/rust-license-ecc269ef87fd38a1.txt` | https://github.com/indexmap-rs/indexmap |
| indicatif | 0.17.11 | MIT | `THIRD_PARTY_LICENSES/rust-license-3e1de3c527ab2512.txt` | https://github.com/console-rs/indicatif |
| ipnet | 2.12.1 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-47dc9ff29128ddfb.txt` | https://github.com/krisprice/ipnet |
| is_terminal_polyfill | 1.70.2 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-6efb0476a1cc0850.txt` | https://github.com/polyfill-rs/is_terminal_polyfill |
| itertools | 0.11.0 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-7576269ea71f767b.txt` | https://github.com/rust-itertools/itertools |
| itertools | 0.12.1 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-7576269ea71f767b.txt` | https://github.com/rust-itertools/itertools |
| itoa | 1.0.18 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-23f18e03dc49df91.txt` | https://github.com/dtolnay/itoa |
| lazy_static | 1.5.0 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-0621878e61f0d0fd.txt` | https://github.com/rust-lang-nursery/lazy-static.rs |
| libc | 0.2.189 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-123a331b5dbf04c3.txt` | https://github.com/rust-lang/libc |
| link-cplusplus | 1.0.12 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-23f18e03dc49df91.txt` | https://github.com/dtolnay/link-cplusplus |
| litemap | 0.8.2 | Unicode-3.0 | `THIRD_PARTY_LICENSES/rust-license-f367c1b8e1aa2624.txt` | https://github.com/unicode-org/icu4x |
| llguidance | 1.7.6 | MIT | `THIRD_PARTY_LICENSES/rust-license-3d4ada4e04d153d7.txt` | https://github.com/guidance-ai/llguidance |
| lock_api | 0.4.14 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-c9a75f18b9ab2927.txt` | https://github.com/Amanieu/parking_lot |
| log | 0.4.33 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-6485b8ed310d3f03.txt` | https://github.com/rust-lang/log |
| macro_rules_attribute | 0.2.3 | Apache-2.0 OR MIT OR Zlib | `THIRD_PARTY_LICENSES/rust-license-603fb27ef3266ea5.txt` | https://github.com/danielhenrymantilla/macro_rules_attribute-rs |
| macro_rules_attribute-proc_macro | 0.2.3 | Apache-2.0 OR MIT OR Zlib | `THIRD_PARTY_LICENSES/rust-license-603fb27ef3266ea5.txt` | https://github.com/danielhenrymantilla/macro_rules_attribute-rs |
| matchers | 0.2.0 | MIT | `THIRD_PARTY_LICENSES/rust-license-a47129d738752a6a.txt` | https://github.com/hawkw/matchers |
| matchit | 0.7.3 | MIT AND BSD-3-Clause | `THIRD_PARTY_LICENSES/rust-license-162ce11ad71338d0.txt`<br>`THIRD_PARTY_LICENSES/rust-license-de701d0618d694fe.txt` | https://github.com/ibraheemdev/matchit |
| memchr | 2.8.3 | Unlicense OR MIT | `THIRD_PARTY_LICENSES/rust-license-0f96a83840e146e4.txt` | https://github.com/BurntSushi/memchr |
| memo-map | 0.3.3 | Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-143368af9701a24e.txt` | https://github.com/mitsuhiko/memo-map |
| mime | 0.3.17 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-8b87502eddb2d7fa.txt` | https://github.com/hyperium/mime |
| minijinja | 2.21.0 | Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-143368af9701a24e.txt` | https://github.com/mitsuhiko/minijinja |
| minimal-lexical | 0.2.1 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-23f18e03dc49df91.txt` | https://github.com/Alexhuszagh/minimal-lexical |
| miniz_oxide | 0.8.9 | MIT OR Zlib OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-799e9ca9d179295e.txt` | https://github.com/Frommi/miniz_oxide/tree/master/miniz_oxide |
| mio | 1.2.2 | MIT | `THIRD_PARTY_LICENSES/rust-license-07919255c7e04793.txt` | https://github.com/tokio-rs/mio |
| monostate | 0.1.18 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-23f18e03dc49df91.txt` | https://github.com/dtolnay/monostate |
| monostate-impl | 0.1.18 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-23f18e03dc49df91.txt` | https://github.com/dtolnay/monostate |
| moxcms | 0.8.1 | BSD-3-Clause OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-eefdaaf8f4ef07e9.txt` | https://github.com/awxkee/moxcms.git |
| nom | 7.1.3 | MIT | `THIRD_PARTY_LICENSES/rust-license-4dbda04344456f09.txt` | https://github.com/Geal/nom |
| nu-ansi-term | 0.50.3 | MIT | `THIRD_PARTY_LICENSES/rust-license-315fde9fe60c6530.txt` | https://github.com/nushell/nu-ansi-term |
| num-traits | 0.2.19 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-6485b8ed310d3f03.txt` | https://github.com/rust-num/num-traits |
| num_cpus | 1.17.0 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-1626f2c950cee975.txt` | https://github.com/seanmonstar/num_cpus |
| number_prefix | 0.4.0 | MIT | `THIRD_PARTY_LICENSES/rust-license-b05785f9f18e6716.txt` | https://github.com/ogham/rust-number-prefix |
| once_cell | 1.21.4 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-23f18e03dc49df91.txt` | https://github.com/matklad/once_cell |
| onig | 6.5.3 | MIT | `THIRD_PARTY_LICENSES/rust-license-e9c1238c5beb73c6.txt` | https://github.com/iwillspeak/rust-onig |
| onig_sys | 69.9.3 | MIT | `THIRD_PARTY_LICENSES/rust-license-71f321038b088358.txt` | https://github.com/rust-onig/rust-onig |
| option-ext | 0.2.0 | MPL-2.0 | `THIRD_PARTY_LICENSES/rust-license-66a3107d5ad6a058.txt` | https://github.com/soc/option-ext.git |
| parking | 2.2.1 | Apache-2.0 OR MIT | `THIRD_PARTY_LICENSES/rust-license-23f18e03dc49df91.txt` | https://github.com/smol-rs/parking |
| parking_lot | 0.12.5 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-c9a75f18b9ab2927.txt` | https://github.com/Amanieu/parking_lot |
| parking_lot_core | 0.9.12 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-c9a75f18b9ab2927.txt` | https://github.com/Amanieu/parking_lot |
| paste | 1.0.15 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-23f18e03dc49df91.txt` | https://github.com/dtolnay/paste |
| pastey | 0.2.3 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-23f18e03dc49df91.txt` | https://github.com/as1100k/pastey |
| percent-encoding | 2.3.2 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-b38f11f6096706e6.txt` | https://github.com/servo/rust-url/ |
| pin-project-lite | 0.2.17 | Apache-2.0 OR MIT | `THIRD_PARTY_LICENSES/rust-license-23f18e03dc49df91.txt` | https://github.com/taiki-e/pin-project-lite |
| piper | 0.2.5 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-23f18e03dc49df91.txt` | https://github.com/smol-rs/piper |
| pkg-config | 0.3.33 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-378f5840b258e277.txt` | https://github.com/rust-lang/pkg-config-rs |
| png | 0.18.1 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-eaf40297c75da471.txt` | https://github.com/image-rs/image-png |
| portable-atomic | 1.14.0 | Apache-2.0 OR MIT | `THIRD_PARTY_LICENSES/rust-license-23f18e03dc49df91.txt` | https://github.com/taiki-e/portable-atomic |
| potential_utf | 0.1.5 | Unicode-3.0 | `THIRD_PARTY_LICENSES/rust-license-f367c1b8e1aa2624.txt` | https://github.com/unicode-org/icu4x |
| ppv-lite86 | 0.2.21 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-4cada0bd02ea3692.txt` | https://github.com/cryptocorrosion/cryptocorrosion |
| proc-macro2 | 1.0.107 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-23f18e03dc49df91.txt` | https://github.com/dtolnay/proc-macro2 |
| pxfm | 0.1.30 | BSD-3-Clause OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-eefdaaf8f4ef07e9.txt` | https://github.com/awxkee/pxfm |
| quick-error | 2.0.1 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-058f01fe181608d0.txt` | http://github.com/tailhook/quick-error |
| quote | 1.0.47 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-23f18e03dc49df91.txt` | https://github.com/dtolnay/quote |
| rand | 0.8.7 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-209fbbe0ad52d923.txt` | https://github.com/rust-random/rand |
| rand | 0.9.5 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-209fbbe0ad52d923.txt` | https://github.com/rust-random/rand |
| rand_chacha | 0.3.1 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-209fbbe0ad52d923.txt` | https://github.com/rust-random/rand |
| rand_chacha | 0.9.0 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-209fbbe0ad52d923.txt` | https://github.com/rust-random/rand |
| rand_core | 0.6.4 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-209fbbe0ad52d923.txt` | https://github.com/rust-random/rand |
| rand_core | 0.9.5 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-209fbbe0ad52d923.txt` | https://github.com/rust-random/rand |
| rayon | 1.12.0 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-0621878e61f0d0fd.txt` | https://github.com/rayon-rs/rayon |
| rayon-cond | 0.3.0 | Apache-2.0 OR MIT | `THIRD_PARTY_LICENSES/rust-license-27995d58ad5c1145.txt` | https://github.com/cuviper/rayon-cond |
| rayon-core | 1.13.0 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-0621878e61f0d0fd.txt` | https://github.com/rayon-rs/rayon |
| ref-cast | 1.0.26 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-23f18e03dc49df91.txt` | https://github.com/dtolnay/ref-cast |
| ref-cast-impl | 1.0.26 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-23f18e03dc49df91.txt` | https://github.com/dtolnay/ref-cast |
| referencing | 0.29.1 | MIT | `THIRD_PARTY_LICENSES/rust-license-a573f030c2ae7eab.txt` | https://github.com/Stranger6667/jsonschema |
| regex | 1.13.1 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-6485b8ed310d3f03.txt` | https://github.com/rust-lang/regex |
| regex-automata | 0.4.16 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-6485b8ed310d3f03.txt` | https://github.com/rust-lang/regex |
| regex-syntax | 0.8.11 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-6485b8ed310d3f03.txt` | https://github.com/rust-lang/regex |
| reqwest | 0.12.28 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-34f17d9067385431.txt` | https://github.com/seanmonstar/reqwest |
| ring | 0.17.14 | Apache-2.0 AND ISC | `THIRD_PARTY_LICENSES/rust-license-143368af9701a24e.txt`<br>`THIRD_PARTY_LICENSES/rust-license-f025ccfb7dfb6bdf.txt` | https://github.com/briansmith/ring |
| rustls | 0.23.43 | Apache-2.0 OR ISC OR MIT | `THIRD_PARTY_LICENSES/rust-license-709e3175b4212f7b.txt` | https://github.com/rustls/rustls |
| rustls-pemfile | 2.2.0 | Apache-2.0 OR ISC OR MIT | `THIRD_PARTY_LICENSES/rust-license-709e3175b4212f7b.txt` | https://github.com/rustls/pemfile |
| rustls-pki-types | 1.15.1 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-9117d922e6671255.txt` | https://github.com/rustls/pki-types |
| rustls-webpki | 0.103.13 | ISC | `THIRD_PARTY_LICENSES/rust-license-5b698ca13897be3a.txt` | https://github.com/rustls/webpki |
| rustversion | 1.0.23 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-23f18e03dc49df91.txt` | https://github.com/dtolnay/rustversion |
| ryu | 1.0.23 | Apache-2.0 OR BSL-1.0 | `THIRD_PARTY_LICENSES/rust-license-074e6e32c86a4c0e.txt` | https://github.com/dtolnay/ryu |
| scopeguard | 1.2.0 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-fb77f0a9c53e473a.txt` | https://github.com/bluss/scopeguard |
| scratch | 1.0.9 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-23f18e03dc49df91.txt` | https://github.com/dtolnay/scratch |
| serde | 1.0.229 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-23f18e03dc49df91.txt` | https://github.com/serde-rs/serde |
| serde_core | 1.0.229 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-23f18e03dc49df91.txt` | https://github.com/serde-rs/serde |
| serde_derive | 1.0.229 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-23f18e03dc49df91.txt` | https://github.com/serde-rs/serde |
| serde_json | 1.0.151 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-23f18e03dc49df91.txt` | https://github.com/serde-rs/json |
| serde_path_to_error | 0.1.20 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-23f18e03dc49df91.txt` | https://github.com/dtolnay/path-to-error |
| serde_urlencoded | 0.7.1 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-b9eb266294324f67.txt` | https://github.com/nox/serde_urlencoded |
| sha2 | 0.10.9 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-b4eb00df6e2a4d22.txt` | https://github.com/RustCrypto/hashes |
| sharded-slab | 0.1.7 | MIT | `THIRD_PARTY_LICENSES/rust-license-eafbfa606bc005ed.txt` | https://github.com/hawkw/sharded-slab |
| shlex | 2.0.1 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-4455bf75a9115410.txt` | https://github.com/comex/rust-shlex |
| simd-adler32 | 0.3.10 | MIT | `THIRD_PARTY_LICENSES/rust-license-42a35170233e83e1.txt` | https://github.com/mcountryman/simd-adler32 |
| slab | 0.4.12 | MIT | `THIRD_PARTY_LICENSES/rust-license-8ce0830173fdac60.txt` | https://github.com/tokio-rs/slab |
| smallvec | 1.15.2 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-0b28172679e0009b.txt` | https://github.com/servo/rust-smallvec |
| socket2 | 0.6.5 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-378f5840b258e277.txt` | https://github.com/rust-lang/socket2 |
| spm_precompiled | 0.1.4 | Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-50e6751797c50ded.txt` | https://github.com/huggingface/spm_precompiled |
| stable_deref_trait | 1.2.1 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-3c125f249fc6fb19.txt` | https://github.com/storyyeller/stable_deref_trait |
| strsim | 0.11.1 | MIT | `THIRD_PARTY_LICENSES/rust-license-1e697ce8d21401fb.txt` | https://github.com/rapidfuzz/strsim-rs |
| strum | 0.28.0 | MIT | `THIRD_PARTY_LICENSES/rust-license-8bce3b45e49ecd14.txt` | https://github.com/Peternator7/strum |
| strum_macros | 0.28.0 | MIT | `THIRD_PARTY_LICENSES/rust-license-8bce3b45e49ecd14.txt` | https://github.com/Peternator7/strum |
| subtle | 2.6.1 | BSD-3-Clause | `THIRD_PARTY_LICENSES/rust-license-cc0332a88c2ea21d.txt` | https://github.com/dalek-cryptography/subtle |
| syn | 2.0.119 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-23f18e03dc49df91.txt` | https://github.com/dtolnay/syn |
| syn | 3.0.3 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-23f18e03dc49df91.txt` | https://github.com/dtolnay/syn |
| sync_wrapper | 1.0.2 | Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-074e6e32c86a4c0e.txt` | https://github.com/Actyx/sync_wrapper |
| synstructure | 0.13.2 | MIT | `THIRD_PARTY_LICENSES/rust-license-219920e865eee70b.txt` | https://github.com/mystor/synstructure |
| system-configuration | 0.7.0 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-000cbb5bea6f2182.txt` | https://github.com/mullvad/system-configuration-rs |
| system-configuration-sys | 0.6.0 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-000cbb5bea6f2182.txt` | https://github.com/mullvad/system-configuration-rs |
| termcolor | 1.4.1 | Unlicense OR MIT | `THIRD_PARTY_LICENSES/rust-license-0f96a83840e146e4.txt` | https://github.com/BurntSushi/termcolor |
| thiserror | 1.0.69 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-23f18e03dc49df91.txt` | https://github.com/dtolnay/thiserror |
| thiserror | 2.0.19 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-23f18e03dc49df91.txt` | https://github.com/dtolnay/thiserror |
| thiserror-impl | 1.0.69 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-23f18e03dc49df91.txt` | https://github.com/dtolnay/thiserror |
| thiserror-impl | 2.0.19 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-23f18e03dc49df91.txt` | https://github.com/dtolnay/thiserror |
| thread_local | 1.1.10 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-c9a75f18b9ab2927.txt` | https://github.com/Amanieu/thread_local-rs |
| tinystr | 0.8.3 | Unicode-3.0 | `THIRD_PARTY_LICENSES/rust-license-f367c1b8e1aa2624.txt` | https://github.com/unicode-org/icu4x |
| tokenizers | 0.20.4 | Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-50e6751797c50ded.txt` | https://github.com/huggingface/tokenizers |
| tokio | 1.53.1 | MIT | `THIRD_PARTY_LICENSES/rust-license-253cd04c6714889d.txt` | https://github.com/tokio-rs/tokio |
| tokio-macros | 2.7.2 | MIT | `THIRD_PARTY_LICENSES/rust-license-0b83dc40cba89b99.txt` | https://github.com/tokio-rs/tokio |
| tokio-rustls | 0.26.4 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-e20fa2b8e0a2565f.txt` | https://github.com/rustls/tokio-rustls |
| tokio-stream | 0.1.19 | MIT | `THIRD_PARTY_LICENSES/rust-license-253cd04c6714889d.txt` | https://github.com/tokio-rs/tokio |
| tokio-util | 0.7.19 | MIT | `THIRD_PARTY_LICENSES/rust-license-253cd04c6714889d.txt` | https://github.com/tokio-rs/tokio |
| toktrie | 1.7.6 | MIT | `THIRD_PARTY_LICENSES/rust-license-3d4ada4e04d153d7.txt` | https://github.com/guidance-ai/llguidance |
| tower | 0.5.3 | MIT | `THIRD_PARTY_LICENSES/rust-license-4249c8e6c5ebb85f.txt` | https://github.com/tower-rs/tower |
| tower-http | 0.6.11 | MIT | `THIRD_PARTY_LICENSES/rust-license-5049cf464977eff4.txt` | https://github.com/tower-rs/tower-http |
| tower-layer | 0.3.3 | MIT | `THIRD_PARTY_LICENSES/rust-license-4249c8e6c5ebb85f.txt` | https://github.com/tower-rs/tower |
| tower-service | 0.3.3 | MIT | `THIRD_PARTY_LICENSES/rust-license-4249c8e6c5ebb85f.txt` | https://github.com/tower-rs/tower |
| tracing | 0.1.44 | MIT | `THIRD_PARTY_LICENSES/rust-license-898b1ae9821e98da.txt` | https://github.com/tokio-rs/tracing |
| tracing-attributes | 0.1.31 | MIT | `THIRD_PARTY_LICENSES/rust-license-898b1ae9821e98da.txt` | https://github.com/tokio-rs/tracing |
| tracing-core | 0.1.36 | MIT | `THIRD_PARTY_LICENSES/rust-license-898b1ae9821e98da.txt` | https://github.com/tokio-rs/tracing |
| tracing-log | 0.2.0 | MIT | `THIRD_PARTY_LICENSES/rust-license-898b1ae9821e98da.txt` | https://github.com/tokio-rs/tracing |
| tracing-subscriber | 0.3.23 | MIT | `THIRD_PARTY_LICENSES/rust-license-898b1ae9821e98da.txt` | https://github.com/tokio-rs/tracing |
| try-lock | 0.2.5 | MIT | `THIRD_PARTY_LICENSES/rust-license-8b62775bacdfa5ae.txt` | https://github.com/seanmonstar/try-lock |
| typenum | 1.20.1 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-a825bd853ab71619.txt` | https://github.com/paholg/typenum |
| unicode-ident | 1.0.24 | (MIT OR Apache-2.0) AND Unicode-3.0 | `THIRD_PARTY_LICENSES/rust-license-23f18e03dc49df91.txt`<br>`THIRD_PARTY_LICENSES/rust-license-f7db81051789b729.txt` | https://github.com/dtolnay/unicode-ident |
| unicode-normalization-alignments | 0.1.12 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-7b63ecd5f1902af1.txt` | https://github.com/n1t0/unicode-normalization |
| unicode-segmentation | 1.13.3 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-7b63ecd5f1902af1.txt` | https://github.com/unicode-rs/unicode-segmentation |
| unicode-width | 0.2.2 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-7b63ecd5f1902af1.txt` | https://github.com/unicode-rs/unicode-width |
| unicode_categories | 0.1.1 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-98a817e7b85e5fe4.txt` | https://github.com/swgillespie/unicode-categories |
| untrusted | 0.9.0 | ISC | `THIRD_PARTY_LICENSES/rust-license-7abd9b6960dcf7d4.txt` | https://github.com/briansmith/untrusted |
| url | 2.5.8 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-b38f11f6096706e6.txt` | https://github.com/servo/rust-url |
| utf8_iter | 1.0.4 | Apache-2.0 OR MIT | `THIRD_PARTY_LICENSES/rust-license-3fa4ca83dcc92378.txt` | https://github.com/hsivonen/utf8_iter |
| utf8parse | 0.2.2 | Apache-2.0 OR MIT | `THIRD_PARTY_LICENSES/rust-license-e4c9b06fa850cb9b.txt` | https://github.com/alacritty/vte |
| uuid | 1.24.0 | Apache-2.0 OR MIT | `THIRD_PARTY_LICENSES/rust-license-436bc5a105d8e57d.txt` | https://github.com/uuid-rs/uuid |
| version_check | 0.9.5 | MIT OR Apache-2.0 | `THIRD_PARTY_LICENSES/rust-license-b7e650f3fce5c532.txt` | https://github.com/SergioBenitez/version_check |
| want | 0.3.1 | MIT | `THIRD_PARTY_LICENSES/rust-license-96d741569b18c610.txt` | https://github.com/seanmonstar/want |
| webpki-roots | 1.0.9 | CDLA-Permissive-2.0 | `THIRD_PARTY_LICENSES/rust-license-e271993808fec50a.txt` | https://github.com/rustls/webpki-roots |
| writeable | 0.6.3 | Unicode-3.0 | `THIRD_PARTY_LICENSES/rust-license-f367c1b8e1aa2624.txt` | https://github.com/unicode-org/icu4x |
| yoke | 0.8.3 | Unicode-3.0 | `THIRD_PARTY_LICENSES/rust-license-f367c1b8e1aa2624.txt` | https://github.com/unicode-org/icu4x |
| yoke-derive | 0.8.2 | Unicode-3.0 | `THIRD_PARTY_LICENSES/rust-license-f367c1b8e1aa2624.txt` | https://github.com/unicode-org/icu4x |
| zerocopy | 0.8.55 | BSD-2-Clause OR Apache-2.0 OR MIT | `THIRD_PARTY_LICENSES/rust-license-24fa231567ace7e0.txt` | https://github.com/google/zerocopy |
| zerocopy-derive | 0.8.55 | BSD-2-Clause OR Apache-2.0 OR MIT | `THIRD_PARTY_LICENSES/rust-license-24fa231567ace7e0.txt` | https://github.com/google/zerocopy |
| zerofrom | 0.1.8 | Unicode-3.0 | `THIRD_PARTY_LICENSES/rust-license-f367c1b8e1aa2624.txt` | https://github.com/unicode-org/icu4x |
| zerofrom-derive | 0.1.7 | Unicode-3.0 | `THIRD_PARTY_LICENSES/rust-license-f367c1b8e1aa2624.txt` | https://github.com/unicode-org/icu4x |
| zeroize | 1.9.0 | Apache-2.0 OR MIT | `THIRD_PARTY_LICENSES/rust-license-8c7516d4b27b1e49.txt` | https://github.com/RustCrypto/utils |
| zerotrie | 0.2.4 | Unicode-3.0 | `THIRD_PARTY_LICENSES/rust-license-f367c1b8e1aa2624.txt` | https://github.com/unicode-org/icu4x |
| zerovec | 0.11.6 | Unicode-3.0 | `THIRD_PARTY_LICENSES/rust-license-f367c1b8e1aa2624.txt` | https://github.com/unicode-org/icu4x |
| zerovec-derive | 0.11.3 | Unicode-3.0 | `THIRD_PARTY_LICENSES/rust-license-f367c1b8e1aa2624.txt` | https://github.com/unicode-org/icu4x |
| zmij | 1.0.23 | MIT | `THIRD_PARTY_LICENSES/rust-license-23f18e03dc49df91.txt` | https://github.com/dtolnay/zmij |
| zune-core | 0.5.1 | MIT OR Apache-2.0 OR Zlib | `THIRD_PARTY_LICENSES/rust-license-d30047bca3b51663.txt` | https://github.com/etemesi254/zune-image |
| zune-jpeg | 0.5.15 | MIT OR Apache-2.0 OR Zlib | `THIRD_PARTY_LICENSES/rust-license-d30047bca3b51663.txt` | https://github.com/etemesi254/zune-image/tree/dev/crates/zune-jpeg |

## Swift dependencies

| Package | Version | Revision | License | License text | Source |
|---|---|---|---|---|---|
| Sparkle | 2.9.6 | `ac2def288cbff5cfc7df3ffef6abdf45b72bcb0a` | MIT and bundled third-party notices | `THIRD_PARTY_LICENSES/swift-sparkle-license.txt` | https://github.com/sparkle-project/Sparkle.git |
| ZIPFoundation | 0.9.20 | `22787ffb59de99e5dc1fbfe80b19c97a904ad48d` | MIT | `THIRD_PARTY_LICENSES/swift-zipfoundation-license.txt` | https://github.com/weichsel/ZIPFoundation.git |

## Bundled third-party assets

| Asset | Source revision | Copyright | License | License text | Bundled file | Source |
|---|---|---|---|---|---|---|
| Hermes Agent logo | `1706502aa70485440a64127475f780c193784d6d` | Copyright (c) 2025 Nous Research | MIT | `THIRD_PARTY_LICENSES/asset-hermes-agent-mit.txt` | `ironmlx-app/Sources/IronMLXAppCore/Resources/hermes-agent-logo.svg` | https://github.com/NousResearch/hermes-agent/blob/1706502aa70485440a64127475f780c193784d6d/website/static/img/apple-touch-icon.png |
| oh-my-pi logo | `v17.2.12` | Copyright (c) 2025 Mario Zechner; Copyright (c) 2025-2026 Can Bölük | MIT | `THIRD_PARTY_LICENSES/asset-oh-my-pi-mit.txt` | `ironmlx-app/Sources/IronMLXAppCore/Resources/oh-my-pi-logo.svg` | https://github.com/can1357/oh-my-pi/blob/45e12e5bb758198a920c6070e7e64cb33b21beac/assets/icon.svg |

Hermes Agent and oh-my-pi names and logos are used solely to identify
supported third-party integrations. No affiliation or endorsement is
implied. All trademarks remain the property of their respective owners.

## Explicit exclusions

Apple system frameworks are supplied by macOS and are not copied into the
App Bundle. Model weights are downloaded separately by the user and are
outside this App-binary inventory; each model remains subject to its own
license and usage terms.

## Review boundary

The generated materials preserve source license texts and detect dependency
drift. Final license interpretation, attribution review, model-license policy,
CycloneDX SBOM production, and authorization for public distribution remain
P0-8B release gates.
