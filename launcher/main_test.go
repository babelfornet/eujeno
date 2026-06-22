package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestIsVersionRequest(t *testing.T) {
	for _, arg := range []string{"--version", "-V", "version"} {
		if !isVersionRequest([]string{arg}) {
			t.Errorf("isVersionRequest([%q]) should be true", arg)
		}
	}
	if isVersionRequest(nil) {
		t.Error("isVersionRequest(nil) should be false")
	}
	if isVersionRequest([]string{"serve"}) {
		t.Error("isVersionRequest([serve]) should be false")
	}
	// a version flag only counts when it is the first argument
	if isVersionRequest([]string{"serve", "--version"}) {
		t.Error("isVersionRequest must only match the first arg")
	}
}

func TestIsUpdateRequest(t *testing.T) {
	if !isUpdateRequest([]string{"update"}) {
		t.Error("isUpdateRequest([update]) should be true")
	}
	if isUpdateRequest(nil) {
		t.Error("isUpdateRequest(nil) should be false")
	}
	// only the bare top-level command counts, not a flag or a subcommand arg
	if isUpdateRequest([]string{"serve", "update"}) {
		t.Error("isUpdateRequest must only match a lone update arg")
	}
	if isUpdateRequest([]string{"update", "--force"}) {
		t.Error("isUpdateRequest must not match update with extra args")
	}
}

func TestLauncherAssetCoversCurrentPlatform(t *testing.T) {
	if launcherAsset() == "" {
		t.Skip("no launcher asset mapping for this test platform")
	}
}

func TestIsHelpRequest(t *testing.T) {
	for _, arg := range []string{"--help", "-h", "help"} {
		if !isHelpRequest([]string{arg}) {
			t.Errorf("isHelpRequest([%q]) should be true", arg)
		}
	}
	if !isHelpRequest(nil) {
		t.Error("isHelpRequest(nil) (bare invocation) should be true")
	}
	if isHelpRequest([]string{"serve"}) {
		t.Error("isHelpRequest([serve]) should be false")
	}
	// per-command help is the real CLI's job, not the launcher's
	if isHelpRequest([]string{"serve", "--help"}) {
		t.Error("isHelpRequest must only match help as the first arg")
	}
}

func TestVersionLineUsesBuildVar(t *testing.T) {
	old := eujenoVersion
	eujenoVersion = "9.9.9"
	defer func() { eujenoVersion = old }()
	if got := versionLine(); got != "eujeno 9.9.9" {
		t.Errorf("versionLine() = %q, want %q", got, "eujeno 9.9.9")
	}
}

func TestHelpTextMentionsCommonCommands(t *testing.T) {
	h := helpText()
	for _, want := range []string{"serve", "up", "infer", "--version"} {
		if !contains(h, want) {
			t.Errorf("helpText() should mention %q", want)
		}
	}
}

// run(--version) must answer instantly without provisioning a runtime.
func TestRunVersionDoesNotProvision(t *testing.T) {
	home := t.TempDir()
	t.Setenv("EUJENO_HOME", home)
	if err := run([]string{"--version"}); err != nil {
		t.Fatalf("run(--version) error: %v", err)
	}
	if _, err := os.Stat(filepath.Join(home, "runtime")); !os.IsNotExist(err) {
		t.Error("run(--version) must not create the runtime directory")
	}
}

// run(--help) on a machine with no runtime yet must print help, not provision.
func TestRunHelpDoesNotProvisionWhenFresh(t *testing.T) {
	home := t.TempDir()
	t.Setenv("EUJENO_HOME", home)
	if err := run([]string{"--help"}); err != nil {
		t.Fatalf("run(--help) error: %v", err)
	}
	if _, err := os.Stat(filepath.Join(home, "runtime", ".installed")); !os.IsNotExist(err) {
		t.Error("run(--help) on a fresh machine must not provision")
	}
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
