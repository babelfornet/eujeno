// eujeno — thin cross-platform launcher.
//
// A tiny native binary (no Python required) that, on first run, bootstraps a
// private runtime: it fetches uv (astral's single-binary Python/dependency
// manager), creates a managed-CPython virtualenv, and installs the eujeno
// wheel with the right torch backend auto-detected (CPU / CUDA / ROCm / MPS).
// After that it just exec's the real eujeno CLI, forwarding all arguments — so
// `eujeno serve --peers <seed> --model <id>` joins a network and
// `eujeno up --model <id>` creates one, exactly like the pip-installed CLI.
//
// Build-time variables (set with -ldflags):
//
//	-X main.eujenoVersion=<version>   the eujeno release this launcher targets
//	-X main.wheelURL=<url>            URL of the eujeno wheel to install
//	-X main.uvVersion=<version>      pinned uv release to fetch
//
// Runtime overrides (handy for testing):
//
//	EUJENO_WHEEL=<path-or-url>        install this wheel instead of the baked URL
//	EUJENO_FORCE_BOOTSTRAP=1          re-create the runtime even if up to date
//	EUJENO_HOME=<dir>                 runtime location (default ~/.eujeno)
package main

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
)

var (
	eujenoVersion = "dev"
	wheelURL      = ""
	uvVersion     = "latest"
)

// repo is the GitHub repository releases are published to.
const repo = "babelfornet/eujeno"

const pythonVersion = "3.12"

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "eujeno launcher error:", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	// `--version` is answered by the launcher itself — the version is baked in
	// at build time, so there is no need to provision the Python runtime.
	if isVersionRequest(args) {
		fmt.Println(versionLine())
		return nil
	}

	// `eujeno update` is handled by the launcher: it replaces this binary with
	// the latest release, and the new runtime is provisioned on the next run.
	if isUpdateRequest(args) {
		return selfUpdate()
	}

	rt := runtimeDir()
	venv := filepath.Join(rt, "venv")
	uvPath := filepath.Join(rt, uvBin())
	stamp := filepath.Join(rt, ".installed")

	provisioned := os.Getenv("EUJENO_FORCE_BOOTSTRAP") == "" && upToDate(stamp)

	// Before the runtime exists, answer `--help` (and a bare invocation) with a
	// concise launcher help rather than provisioning Python just to print it.
	// Once provisioned, help falls through to the real CLI for the full,
	// per-command output.
	if !provisioned && isHelpRequest(args) {
		fmt.Print(helpText())
		return nil
	}

	if !provisioned {
		fmt.Fprintln(os.Stderr, "eujeno: first-run setup (one time) — provisioning Python + torch…")
		if err := bootstrap(rt, venv, uvPath); err != nil {
			return err
		}
		if err := os.WriteFile(stamp, []byte(installID()), 0o644); err != nil {
			return err
		}
		fmt.Fprintln(os.Stderr, "eujeno: ready.")
	}
	return execEujeno(venv, args)
}

// isVersionRequest reports whether the launcher should print its version and
// exit — true only when a version flag is the first argument (so that, e.g.,
// `eujeno serve --version` is left to the real CLI).
func isVersionRequest(args []string) bool {
	if len(args) == 0 {
		return false
	}
	switch args[0] {
	case "--version", "-V", "version":
		return true
	}
	return false
}

// isHelpRequest reports whether this is a top-level help request (a bare
// invocation, or a help flag as the first argument). Per-command help such as
// `eujeno serve --help` is left to the real CLI.
func isHelpRequest(args []string) bool {
	if len(args) == 0 {
		return true
	}
	switch args[0] {
	case "--help", "-h", "help":
		return true
	}
	return false
}

// isUpdateRequest reports whether this is the top-level `eujeno update` command.
func isUpdateRequest(args []string) bool {
	return len(args) == 1 && args[0] == "update"
}

func versionLine() string {
	return "eujeno " + eujenoVersion
}

// ── self-update ─────────────────────────────────────────────────────────────

// selfUpdate replaces the running launcher binary with the latest release for
// this OS/arch. The new launcher carries a new baked version, so on the next
// command it sees the runtime is stale and re-provisions the new wheel itself.
func selfUpdate() error {
	asset := launcherAsset()
	if asset == "" {
		return fmt.Errorf("unsupported platform %s/%s — update manually from the releases page", runtime.GOOS, runtime.GOARCH)
	}
	self, err := os.Executable()
	if err != nil {
		return err
	}
	if resolved, err := filepath.EvalSymlinks(self); err == nil {
		self = resolved
	}
	url := fmt.Sprintf("https://github.com/%s/releases/latest/download/%s", repo, asset)
	tmp := self + ".new"
	fmt.Fprintf(os.Stderr, "eujeno: fetching the latest launcher (%s)…\n", asset)
	if err := download(url, tmp); err != nil {
		return fmt.Errorf("download %s: %w", url, err)
	}
	if err := os.Chmod(tmp, 0o755); err != nil {
		os.Remove(tmp)
		return err
	}
	newVer := binVersion(tmp)
	if newVer != "" && newVer == eujenoVersion {
		os.Remove(tmp)
		fmt.Fprintf(os.Stderr, "eujeno: already up to date (%s).\n", eujenoVersion)
		return nil
	}
	if err := replaceExe(self, tmp); err != nil {
		os.Remove(tmp)
		return fmt.Errorf("replace %s: %w (try re-running with elevated permissions, or reinstall via https://eujeno.com/install.sh)", self, err)
	}
	if newVer == "" {
		newVer = "the latest version"
	}
	fmt.Fprintf(os.Stderr, "eujeno: updated %s → %s. The next command will provision the new runtime.\n", eujenoVersion, newVer)
	return nil
}

// launcherAsset is the release asset name of the launcher for this OS/arch.
func launcherAsset() string {
	switch runtime.GOOS + "/" + runtime.GOARCH {
	case "darwin/arm64":
		return "eujeno-macos-arm64"
	case "darwin/amd64":
		return "eujeno-macos-x64"
	case "linux/amd64":
		return "eujeno-linux-x64"
	case "linux/arm64":
		return "eujeno-linux-arm64"
	case "windows/amd64":
		return "eujeno-windows-x64.exe"
	}
	return ""
}

// binVersion runs `<path> --version` and returns the version it prints, or "".
func binVersion(path string) string {
	out, err := exec.Command(path, "--version").Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(string(out)), "eujeno"))
}

// replaceExe swaps the running binary at self for the freshly downloaded tmp.
func replaceExe(self, tmp string) error {
	if runtime.GOOS == "windows" {
		// A running .exe can't be overwritten, but it can be renamed: move the
		// old one aside, then the new one into place.
		old := self + ".old"
		os.Remove(old)
		if err := os.Rename(self, old); err != nil {
			return err
		}
		if err := os.Rename(tmp, self); err != nil {
			os.Rename(old, self) // roll back
			return err
		}
		os.Remove(old) // best-effort; may be locked until this process exits
		return nil
	}
	// On Unix, renaming over the running binary is fine — the process keeps the
	// old inode, the next launch uses the new file.
	return os.Rename(tmp, self)
}

func helpText() string {
	return "eujeno " + eujenoVersion + " — decentralized peer-to-peer LLM inference\n" + `
The first real command provisions a private Python runtime (one time);
version and help are answered instantly, without it.

Usage:
  eujeno <command> [options]

Common commands:
  up     --model <id>                          start a coordinator + a node covering the whole model
  serve  --peers <url> --model <id>            run a node and join an existing network
  infer  --coordinator <url> --prompt "..."    query the distributed model
  models                                        list compatible models
  ui     --node <url>                           open a node's dashboard
  update                                        replace this launcher with the latest release

Flags:
  --version, -V    print the eujeno version
  --help, -h       show this help (full per-command help is available after first run)

Every command also accepts --json for machine-readable output.
Docs: https://eujeno.com/docs
`
}

// installID identifies the currently-provisioned runtime so we know when to
// re-bootstrap (new launcher version or new wheel).
func installID() string {
	w := wheelURL
	if env := os.Getenv("EUJENO_WHEEL"); env != "" {
		w = env
	}
	return eujenoVersion + "\n" + w + "\n"
}

func upToDate(stamp string) bool {
	b, err := os.ReadFile(stamp)
	return err == nil && string(b) == installID()
}

func bootstrap(rt, venv, uvPath string) error {
	if err := os.MkdirAll(rt, 0o755); err != nil {
		return err
	}
	if !fileExists(uvPath) {
		if err := fetchUV(rt, uvPath); err != nil {
			return fmt.Errorf("fetch uv: %w", err)
		}
	}
	// Managed CPython + venv (uv downloads the interpreter if absent).
	os.RemoveAll(venv)
	if err := runCmd(uvPath, "venv", "--python", pythonVersion, venv); err != nil {
		return fmt.Errorf("create venv: %w", err)
	}
	wheel := wheelURL
	if env := os.Getenv("EUJENO_WHEEL"); env != "" {
		wheel = env
	}
	if wheel == "" {
		return fmt.Errorf("no eujeno wheel configured (build with -X main.wheelURL=... or set EUJENO_WHEEL)")
	}
	// --torch-backend=auto makes uv pick CPU/CUDA/ROCm/MPS for this machine.
	py := venvPython(venv)
	env := append(os.Environ(), "UV_TORCH_BACKEND=auto", "VIRTUAL_ENV="+venv)
	if err := runCmdEnv(env, uvPath, "pip", "install", "--python", py, "--torch-backend=auto", wheel); err != nil {
		return fmt.Errorf("install eujeno: %w", err)
	}
	return nil
}

func execEujeno(venv string, args []string) error {
	bin := filepath.Join(venv, binDir(), exeName("eujeno"))
	if !fileExists(bin) {
		return fmt.Errorf("eujeno not found in runtime (%s); re-run with EUJENO_FORCE_BOOTSTRAP=1", bin)
	}
	cmd := exec.Command(bin, args...)
	cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr
	if err := cmd.Run(); err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			os.Exit(ee.ExitCode())
		}
		return err
	}
	return nil
}

// ── platform helpers ────────────────────────────────────────────────────────

func runtimeDir() string {
	if h := os.Getenv("EUJENO_HOME"); h != "" {
		return filepath.Join(h, "runtime")
	}
	home, _ := os.UserHomeDir()
	if runtime.GOOS == "windows" {
		if base := os.Getenv("LOCALAPPDATA"); base != "" {
			return filepath.Join(base, "eujeno", "runtime")
		}
	}
	return filepath.Join(home, ".eujeno", "runtime")
}

func binDir() string {
	if runtime.GOOS == "windows" {
		return "Scripts"
	}
	return "bin"
}

func venvPython(venv string) string {
	return filepath.Join(venv, binDir(), exeName("python"))
}

func exeName(n string) string {
	if runtime.GOOS == "windows" {
		return n + ".exe"
	}
	return n
}

func uvBin() string { return exeName("uv") }

func fileExists(p string) bool {
	_, err := os.Stat(p)
	return err == nil
}

func runCmd(name string, args ...string) error {
	return runCmdEnv(os.Environ(), name, args...)
}

func runCmdEnv(env []string, name string, args ...string) error {
	cmd := exec.Command(name, args...)
	cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stderr, os.Stderr // logs to stderr; stdout reserved for the node
	cmd.Env = env
	return cmd.Run()
}

// ── uv download/extraction ──────────────────────────────────────────────────

func fetchUV(rt, dest string) error {
	asset, isZip := uvAsset()
	if asset == "" {
		return fmt.Errorf("unsupported platform %s/%s", runtime.GOOS, runtime.GOARCH)
	}
	tag := "latest/download"
	if uvVersion != "" && uvVersion != "latest" {
		tag = "download/" + uvVersion
	}
	url := fmt.Sprintf("https://github.com/astral-sh/uv/releases/%s/%s", tag, asset)
	tmp := filepath.Join(rt, asset)
	if err := download(url, tmp); err != nil {
		return err
	}
	defer os.Remove(tmp)
	if isZip {
		return extractUVZip(tmp, dest)
	}
	return extractUVTarGz(tmp, dest)
}

func uvAsset() (string, bool) {
	switch runtime.GOOS + "/" + runtime.GOARCH {
	case "darwin/arm64":
		return "uv-aarch64-apple-darwin.tar.gz", false
	case "darwin/amd64":
		return "uv-x86_64-apple-darwin.tar.gz", false
	case "linux/amd64":
		return "uv-x86_64-unknown-linux-gnu.tar.gz", false
	case "linux/arm64":
		return "uv-aarch64-unknown-linux-gnu.tar.gz", false
	case "windows/amd64":
		return "uv-x86_64-pc-windows-msvc.zip", true
	}
	return "", false
}

func download(url, dest string) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("GET %s: %s", url, resp.Status)
	}
	f, err := os.Create(dest)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = io.Copy(f, resp.Body)
	return err
}

func extractUVTarGz(archive, dest string) error {
	f, err := os.Open(archive)
	if err != nil {
		return err
	}
	defer f.Close()
	gz, err := gzip.NewReader(f)
	if err != nil {
		return err
	}
	defer gz.Close()
	tr := tar.NewReader(gz)
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
		if hdr.Typeflag == tar.TypeReg && filepath.Base(hdr.Name) == "uv" {
			return writeExe(dest, tr)
		}
	}
	return fmt.Errorf("uv binary not found in %s", archive)
}

func extractUVZip(archive, dest string) error {
	zr, err := zip.OpenReader(archive)
	if err != nil {
		return err
	}
	defer zr.Close()
	for _, zf := range zr.File {
		if filepath.Base(zf.Name) == "uv.exe" {
			rc, err := zf.Open()
			if err != nil {
				return err
			}
			defer rc.Close()
			return writeExe(dest, rc)
		}
	}
	return fmt.Errorf("uv.exe not found in %s", archive)
}

func writeExe(dest string, r io.Reader) error {
	out, err := os.OpenFile(dest, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o755)
	if err != nil {
		return err
	}
	defer out.Close()
	if _, err := io.Copy(out, r); err != nil {
		return err
	}
	return nil
}
