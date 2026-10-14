//go:build cgo

package ort

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestEnvironment(t *testing.T) {
	libPath := os.Getenv("ORT_LIB_PATH")
	if libPath == "" {
		t.Fatal("ORT_LIB_PATH is unset")
	}

	t.Setenv("ORT_LIB_PATH", "")
	if err := Init(); err == nil {
		t.Fatal("Init succeeded without ORT_LIB_PATH")
	}
	if _, err := GetVersion(); err == nil {
		t.Fatal("GetVersion succeeded before initialization")
	}

	t.Setenv("ORT_LIB_PATH", filepath.Join(t.TempDir(), "libonnxruntime-missing"))
	if err := Init(); err == nil {
		t.Fatal("Init succeeded for a library path that does not open")
	}

	t.Setenv("ORT_LIB_PATH", libPath)
	if err := Init(); err != nil {
		t.Fatalf("Init after a failed attempt: %v", err)
	}
	version, err := GetVersion()
	if err != nil {
		t.Fatal(err)
	}
	if version != "1.32.0" {
		t.Fatalf("onnx runtime version %q", version)
	}
	if err := Init(); err != nil {
		t.Fatalf("second Init: %v", err)
	}

	if CPUExecutionProvider != "CPUExecutionProvider" {
		t.Fatalf("production provider %s", CPUExecutionProvider)
	}
	for _, threads := range []int{1, 0} {
		session, openErr := Open(filepath.Join(t.TempDir(), "unused.onnx"), threads, threads, "NoSuchProvider")
		if openErr == nil || session != nil {
			t.Fatalf("threads %d: session=%v err=%v", threads, session, openErr)
		}
		if !strings.Contains(openErr.Error(), "append execution provider") || !strings.Contains(openErr.Error(), "NoSuchProvider") {
			t.Fatalf("threads %d: %v", threads, openErr)
		}
		if strings.Contains(openErr.Error(), "create session") {
			t.Fatalf("threads %d: session was created after the provider was rejected: %v", threads, openErr)
		}
	}

	if err := Shutdown(); err != nil {
		t.Fatalf("Shutdown: %v", err)
	}
	if err := Init(); err == nil {
		t.Fatal("Init succeeded after shutdown")
	}
}

func TestRuntimeModuleLinked(t *testing.T) {
	out := listDeps(t, "1")
	const moduleLine = "github.com/microsoft/onnxruntime/go v0.0.0-20261013071300-9036f076a2ce"
	if !strings.Contains(out, moduleLine) {
		t.Fatalf("cgo deps:\n%s", out)
	}
}

func listDeps(t *testing.T, cgo string) string {
	t.Helper()
	cmd := exec.Command("go", "list", "-deps", "-f", "{{if .Module}}{{.Module.Path}} {{.Module.Version}}{{end}}", ".")
	cmd.Env = withoutCGO(os.Environ())
	cmd.Env = append(cmd.Env, "CGO_ENABLED="+cgo)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("go list -deps: %v\n%s", err, out)
	}
	return string(out)
}

func withoutCGO(env []string) []string {
	out := make([]string, 0, len(env))
	for _, kv := range env {
		if !strings.HasPrefix(kv, "CGO_ENABLED=") {
			out = append(out, kv)
		}
	}
	return out
}
