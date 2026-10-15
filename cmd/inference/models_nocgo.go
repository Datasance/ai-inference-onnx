//go:build !cgo

package main

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/eclipse-iofog/ai-inference-onnx/internal/batch"
	"github.com/eclipse-iofog/ai-inference-onnx/internal/config"
	"github.com/eclipse-iofog/ai-inference-onnx/internal/httpapi"
)

var (
	_ httpapi.Catalog = (*modelRuntime)(nil)
	_ httpapi.Runner  = (*modelRuntime)(nil)
)

// modelRuntime is the set of models loaded for this process.
// This build does not load the runtime library, so the set stays empty and ready stays false.
type modelRuntime struct{}

// startModels leaves the model set empty.
func startModels(config.Config, *slog.Logger) *modelRuntime {
	return &modelRuntime{}
}

// ready reports that no model can serve.
func (m *modelRuntime) ready() bool {
	var states []httpapi.State
	return httpapi.AnyReady(states...)
}

// Models returns no directories. This build does not load the runtime library.
func (m *modelRuntime) Models() []batch.Directory {
	return nil
}

// Run reports that the model was not found. This build does not load the runtime library.
func (m *modelRuntime) Run(_ context.Context, name, _ string, _ *batch.Request) (*batch.Reply, error) {
	return nil, batch.NotFound(fmt.Sprintf("model %q was not found", name))
}

// shutdown releases nothing.
func (m *modelRuntime) shutdown() error {
	return nil
}
