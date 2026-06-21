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
)

var (
	eujenoVersion = "dev"
	wheelURL      = ""
	uvVersion     = "latest"
)

const pythonVersion = "3.12"

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "eujeno launcher error:", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	rt := runtimeDir()
	venv := filepath.Join(rt, "venv")
	uvPath := filepath.Join(rt, uvBin())
	stamp := filepath.Join(rt, ".installed")

	if os.Getenv("EUJENO_FORCE_BOOTSTRAP") != "" || !upToDate(stamp) {
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
