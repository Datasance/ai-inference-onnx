//go:build cgo

package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"

	"github.com/eclipse-iofog/ai-inference-onnx/internal/config"
	"github.com/eclipse-iofog/ai-inference-onnx/internal/httpapi"
	"github.com/eclipse-iofog/ai-inference-onnx/internal/ort"
	"github.com/eclipse-iofog/ai-inference-onnx/internal/session"
)

// modelRuntime is the set of models loaded for this process.
type modelRuntime struct {
	repo    *session.Repository
	watcher *session.Watcher
	started bool
}

// startModels loads the shared library, opens one session per ready model version,
// and watches the model and knowledge directories.
// Thread counts come from server settings. Each session requests the CPU provider.
// A library that cannot be loaded, or a models directory that cannot be read, leaves the set empty.
// The process still answers, and ready stays false until a model can serve.
func startModels(cfg config.Config, logger *slog.Logger) *modelRuntime {
	if err := ort.Init(); err != nil {
		logger.Error(fmt.Sprintf("runtime: %v", err))
		return &modelRuntime{}
	}
	running := &modelRuntime{started: true}
	repo, err := openRepository(cfg, logger)
	if err != nil {
		logger.Error(fmt.Sprintf("models: %v", err))
		return running
	}
	running.repo = repo
	watcher, err := watchRepository(cfg, repo)
	if err != nil {
		logger.Error(fmt.Sprintf("watch: %v", err))
		return running
	}
	running.watcher = watcher
	return running
}

func openRepository(cfg config.Config, logger *slog.Logger) (*session.Repository, error) {
	pool, err := session.NewPool(cfg.MaxPayloadBytes)
	if err != nil {
		return nil, err
	}
	sessCfg := sessionConfig(cfg)
	sessCfg.OnState = func(name string, state session.State) {
		logModelState(logger, name, state)
	}
	return session.Load(sessCfg, pool)
}

func watchRepository(cfg config.Config, repo *session.Repository) (*session.Watcher, error) {
	watchCfg := sessionConfig(cfg)
	// A missing knowledge directory is not watched.
	// Models still load, and a named item that is absent stays unavailable.
	if !directory(cfg.KnowledgePath) {
		watchCfg.KnowledgeRoot = ""
	}
	return session.Watch(context.Background(), repo, watchCfg, session.PollPeriod, session.QuietPeriod)
}

func sessionConfig(cfg config.Config) session.Config {
	return session.Config{
		ModelsRoot:    cfg.ModelsPath,
		KnowledgeRoot: cfg.KnowledgePath,
		IntraOp:       cfg.IntraOpThreads,
		InterOp:       cfg.InterOpThreads,
		Provider:      ort.CPUExecutionProvider,
	}
}

func directory(path string) bool {
	info, err := os.Stat(path)
	if err != nil {
		return false
	}
	return info.IsDir()
}

// ready reports whether any loaded model can serve.
func (m *modelRuntime) ready() bool {
	if m == nil || m.repo == nil {
		return httpapi.AnyReady()
	}
	listed := m.repo.Models()
	states := make([]httpapi.State, len(listed))
	for i, model := range listed {
		states[i] = httpapi.State(model.State)
	}
	return httpapi.AnyReady(states...)
}

// shutdown stops directory watches, lets an in-flight run finish, closes sessions, and then releases the runtime.
func (m *modelRuntime) shutdown() error {
	if m == nil {
		return nil
	}
	var errs []error
	if m.watcher != nil {
		errs = append(errs, m.watcher.Close())
	}
	if m.repo != nil {
		errs = append(errs, m.repo.Close())
	}
	if m.started {
		errs = append(errs, ort.Shutdown())
	}
	return errors.Join(errs...)
}
