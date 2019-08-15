[![Build Status](https://travis-ci.com/Helga-Helga/stereo-vision.svg?branch=master)](https://travis-ci.com/Helga-Helga/stereo-vision)

# stereo-vision

Implementation of diffusion algorithm for stereo vision.

## How to install Rust

The program was written on
[Rust](https://en.wikipedia.org/wiki/Rust_(programming_language)).
Find language help [here](https://doc.rust-lang.org/stable/book/).

You can find how to install Rust
[here](https://doc.rust-lang.org/stable/book/ch01-01-installation.html).

## How to compile and run
To compile and run the project, type
[`cargo run`](https://doc.rust-lang.org/stable/book/ch01-03-hello-cargo.html)
from the package folder in the terminal.
You can use `cargo build` and `curgo run` separately.
To check if there are some issues in the code without build, run `cargo check`.

To build and run in release mode, use flag `--release`.
It will optimize code, so it is mush faster.

## How to check code coverage
To check code coverage use
[cargo-tarpaulin tool](https://crates.io/crates/cargo-tarpaulin).
