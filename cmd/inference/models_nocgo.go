//go:build !cgo

package main

import (
	"log/slog"

	"github.com/eclipse-iofog/ai-inference-onnx/internal/config"
	"github.com/eclipse-iofog/ai-inference-onnx/internal/httpapi"
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

// shutdown releases nothing.
func (m *modelRuntime) shutdown() error {
	return nil
}
