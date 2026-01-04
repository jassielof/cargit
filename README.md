# Cargit

CLI package tool manager for Git-based Cargo projects.

Unlike `cargo install`, cargit caches repositories, making updates faster.

## Installation

Use Astral's UV tool to install this repository.

### Dependencies

You are expected to have both Git and Rust's Cargo installed and discoverable in your system's shell.

### Cargo configuration

Some cargo configuration is required, specifically setting up the target directory for shared build artifacts. For example `~/.cargo/target`.
