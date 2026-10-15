//go:build cgo

package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"time"

	"github.com/eclipse-iofog/ai-inference-onnx/internal/batch"
	"github.com/eclipse-iofog/ai-inference-onnx/internal/config"
	"github.com/eclipse-iofog/ai-inference-onnx/internal/httpapi"
	"github.com/eclipse-iofog/ai-inference-onnx/internal/ort"
	"github.com/eclipse-iofog/ai-inference-onnx/internal/session"
)

var (
	_ httpapi.Catalog = (*modelRuntime)(nil)
	_ httpapi.Runner  = (*modelRuntime)(nil)
)

// modelRuntime is the set of models loaded for this process.
type modelRuntime struct {
	repo    *session.Repository
	watcher *session.Watcher
	started bool
	queues  inferQueues
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
	repo, pool, err := openRepository(cfg, logger)
	if err != nil {
		logger.Error(fmt.Sprintf("models: %v", err))
		return running
	}
	running.repo = repo
	running.queues.capacity = cfg.QueueCapacity
	running.queues.delayDefault = delayFromMicroseconds(cfg.BatchDelayMicroseconds)
	running.queues.pool = bufferPool{pool: pool}
	running.queues.logger = logger
	running.queues.slot = running.slot
	running.queues.admit = running.admit
	running.queues.settings = running.batchSettings
	watcher, err := watchRepository(cfg, repo)
	if err != nil {
		logger.Error(fmt.Sprintf("watch: %v", err))
		return running
	}
	running.watcher = watcher
	return running
}

func openRepository(cfg config.Config, logger *slog.Logger) (*session.Repository, *session.Pool, error) {
	pool, err := session.NewPool(cfg.MaxPayloadBytes)
	if err != nil {
		return nil, nil, err
	}
	sessCfg := sessionConfig(cfg)
	sessCfg.OnState = func(name string, state session.State) {
		logModelState(logger, name, state)
	}
	repo, err := session.Load(sessCfg, pool)
	if err != nil {
		return nil, nil, err
	}
	return repo, pool, nil
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

// Models returns the directories currently recorded.
// An empty runtime has none.
func (m *modelRuntime) Models() []batch.Directory {
	if m == nil || m.repo == nil {
		return nil
	}
	return directoriesFrom(m.repo.Catalog())
}

func directoriesFrom(models []session.CatalogModel) []batch.Directory {
	if len(models) == 0 {
		return nil
	}
	out := make([]batch.Directory, len(models))
	for i, model := range models {
		versions := make([]batch.ModelVersion, len(model.Versions))
		for j, version := range model.Versions {
			versions[j] = batch.ModelVersion{
				Version: version.Version,
				Signature: batch.Signature{
					Inputs:       tensorSpecs(version.Inputs),
					Outputs:      tensorSpecs(version.Outputs),
					MaxBatchSize: version.MaxBatchSize,
				},
			}
		}
		out[i] = batch.Directory{
			Name:     model.Name,
			State:    string(model.State),
			Reason:   model.Reason,
			Versions: versions,
		}
	}
	return out
}

func tensorSpecs(tensors []session.CatalogTensor) []batch.TensorSpec {
	out := make([]batch.TensorSpec, len(tensors))
	for i, tensor := range tensors {
		shape := append([]int64(nil), tensor.Shape...)
		if len(shape) == 0 {
			shape = []int64{}
		}
		out[i] = batch.TensorSpec{
			Name:     tensor.Name,
			Datatype: batch.Datatype(tensor.Datatype),
			Shape:    shape,
		}
	}
	return out
}

// Run submits one decoded request to the queue of the named version.
// The queue calls the session that is already open for that version.
func (m *modelRuntime) Run(ctx context.Context, name, version string, req *batch.Request) (*batch.Reply, error) {
	if m == nil || m.repo == nil {
		return nil, batch.NotFound(fmt.Sprintf("model %q was not found", name))
	}
	return m.queues.submit(ctx, name, version, req)
}

func (m *modelRuntime) admit(name, version string) error {
	if m == nil || m.repo == nil {
		return batch.NotFound(fmt.Sprintf("model %q was not found", name))
	}
	admission := m.repo.Admit(name, version)
	if admission.Proceed() {
		return nil
	}
	if err := batch.GateError(admission.Status, admission.Reason); err != nil {
		return err
	}
	return batch.NotFound(fmt.Sprintf("model %q was not found", name))
}

func (m *modelRuntime) batchSettings(name, version string) (int, time.Duration, bool) {
	if m == nil || m.repo == nil {
		return 0, 0, false
	}
	for _, model := range m.repo.Catalog() {
		if model.Name != name {
			continue
		}
		delay := m.queues.delayDefault
		if model.QueueDelaySet {
			delay = unsignedDelay(model.QueueDelayMicroseconds)
		}
		for _, item := range model.Versions {
			if item.Version == version {
				return item.MaxBatchSize, delay, true
			}
		}
		return 0, 0, false
	}
	return 0, 0, false
}

func (m *modelRuntime) slot(name, version string) batch.Slot {
	return func(ctx context.Context, inputs []batch.Tensor) ([]batch.Tensor, func(), error) {
		if m == nil || m.repo == nil {
			return nil, nil, batch.NotFound(fmt.Sprintf("model %q was not found", name))
		}
		converted := make([]session.Input, len(inputs))
		for i, in := range inputs {
			converted[i] = session.Input{Name: in.Name, Shape: in.Shape, Data: in.Data}
		}
		outputs, err := m.repo.Run(ctx, name, version, converted)
		if err != nil {
			return nil, nil, runError(err)
		}
		tensors := make([]batch.Tensor, len(outputs))
		for i, out := range outputs {
			tensors[i] = batch.Tensor{Name: out.Name, Shape: out.Shape, Data: out.Buffer.Data}
		}
		release := func() {
			for _, out := range outputs {
				m.repo.Release(out)
			}
		}
		return tensors, release, nil
	}
}

func runError(err error) error {
	var status *session.StatusError
	if errors.As(err, &status) {
		if mapped := batch.GateError(status.Status, status.Reason); mapped != nil {
			return mapped
		}
		return batch.NotFound("model was not found")
	}
	return err
}

// bufferPool adapts the session buffer pool to the batcher.
type bufferPool struct {
	pool *session.Pool
}

func (p bufferPool) Get(n int) ([]byte, error) {
	if p.pool == nil {
		return nil, errors.New("buffer pool is closed")
	}
	buf, err := p.pool.Get(n, nil)
	if err != nil {
		return nil, err
	}
	return buf.Data, nil
}

func (p bufferPool) Put(b []byte) {
	if p.pool == nil || cap(b) == 0 {
		return
	}
	p.pool.Put(session.Buffer{Data: b, Pooled: true})
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
