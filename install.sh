#!/bin/sh
# Eujeno installer — fetch the native launcher for your OS/arch from the latest
# GitHub Release and install it as `eujeno` on your PATH.
#
#   curl -fsSL https://eujeno.com/install.sh | sh
#
# The launcher is a tiny static binary; on first run it provisions a private
# Python runtime (via uv, with the right PyTorch backend) and installs the
# eujeno wheel — nothing else to install.
#
# Environment overrides:
#   EUJENO_VERSION=v0.1.0   install a specific release (default: latest)
#   EUJENO_BIN_DIR=~/bin     install location (default: /usr/local/bin or ~/.local/bin)
set -eu

REPO="babelfornet/eujeno"
VERSION="${EUJENO_VERSION:-latest}"

say() { printf '\033[1;36meujeno\033[0m %s\n' "$1" >&2; }
err() { printf '\033[1;31meujeno: %s\033[0m\n' "$1" >&2; exit 1; }

# --- detect platform ---------------------------------------------------------
os="$(uname -s)"
arch="$(uname -m)"

case "$os" in
  Darwin) os="macos" ;;
  Linux)  os="linux" ;;
  *) err "unsupported OS '$os'. On Windows, download eujeno-windows-x64.exe from https://github.com/$REPO/releases/latest" ;;
esac

case "$arch" in
  x86_64|amd64)  arch="x64" ;;
  arm64|aarch64) arch="arm64" ;;
  *) err "unsupported architecture '$arch'" ;;
esac

asset="eujeno-${os}-${arch}"

if [ "$VERSION" = "latest" ]; then
  url="https://github.com/$REPO/releases/latest/download/$asset"
else
  url="https://github.com/$REPO/releases/download/$VERSION/$asset"
fi

# --- pick an install dir -----------------------------------------------------
if [ -n "${EUJENO_BIN_DIR:-}" ]; then
  bindir="$EUJENO_BIN_DIR"
elif [ -w /usr/local/bin ]; then
  bindir="/usr/local/bin"
else
  bindir="$HOME/.local/bin"
fi
mkdir -p "$bindir"

# --- download ----------------------------------------------------------------
tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT INT TERM

say "downloading $asset ($VERSION)…"
if command -v curl >/dev/null 2>&1; then
  curl -fsSL "$url" -o "$tmp" || err "download failed: $url"
elif command -v wget >/dev/null 2>&1; then
  wget -qO "$tmp" "$url" || err "download failed: $url"
else
  err "need curl or wget to download"
fi

[ -s "$tmp" ] || err "downloaded an empty file from $url"

chmod +x "$tmp"
mv "$tmp" "$bindir/eujeno"
trap - EXIT INT TERM

say "installed → $bindir/eujeno"

# --- PATH hint ---------------------------------------------------------------
case ":$PATH:" in
  *":$bindir:"*) ;;
  *)
    say "note: $bindir is not on your PATH. Add it with:"
    printf '         echo '\''export PATH="%s:$PATH"'\'' >> ~/.profile && . ~/.profile\n' "$bindir" >&2
    ;;
esac

say "done. Run 'eujeno --help' to get started (first run provisions Python + PyTorch)."
