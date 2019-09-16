[![Build Status](https://travis-ci.com/Helga-Helga/stereo-vision.svg?branch=master)](https://travis-ci.com/Helga-Helga/stereo-vision)
[![Coverage Status](https://coveralls.io/repos/github/Helga-Helga/stereo-vision/badge.svg?branch=master)](https://coveralls.io/github/Helga-Helga/stereo-vision?branch=master)

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
You can use `cargo build` and `cargo run` separately.
To check if there are some issues in the code without build, run `cargo check`.

To build and run in release mode, use flag `--release`.
It will optimize code, so it is mush faster.

## How to check code coverage

To check code coverage use
[cargo-tarpaulin tool](https://crates.io/crates/cargo-tarpaulin).

## How to generate documentation

To generate documentation use [`cargo doc`](https://doc.rust-lang.org/rustdoc/what-is-rustdoc.html).

## Image format

[PMG](http://davis.lbl.gov/Manuals/NETPBM/doc/pgm.html) format is used for input and output images. Images can be only grayscale.

Image file structure:

```
P2
width height
The maximum gray value
Matrix representing pixels intensities
```

where `P2` is a magic number identifying the file type.
