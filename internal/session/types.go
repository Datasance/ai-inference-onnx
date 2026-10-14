package session

// State is the load state of one model directory.
type State string

const (
	// Ready means the model can run.
	Ready State = "READY"
	// Loading means a session is being opened.
	Loading State = "LOADING"
	// Unavailable means the model is known and cannot run.
	Unavailable State = "UNAVAILABLE"
)

const (
	// StatusLoading is the call-gate status while a model is loading.
	StatusLoading = 503
	// StatusMissing is the call-gate status for an unknown name or a version that was not selected.
	StatusMissing = 404
)

const reasonLoading = "model is loading"

// StateFunc is called after a model state is stored.
// name is the model directory. state is the stored state.
// A nil function does nothing. The function must not call back into the repository.
type StateFunc func(name string, state State)

// Config chooses directories, thread counts, the execution provider, and an optional state function for a load.
type Config struct {
	// ModelsRoot is the directory whose children are models.
	ModelsRoot string
	// KnowledgeRoot is the directory that holds named knowledge items.
	KnowledgeRoot string
	// IntraOp is the intra-op thread count passed to each session, including zero.
	IntraOp int
	// InterOp is the inter-op thread count passed to each session, including zero.
	InterOp int
	// Provider is the only execution provider requested for this load.
	// Production callers pass CPUExecutionProvider.
	Provider string
	// OnState is called after a model state is stored.
	// A nil function does nothing.
	OnState StateFunc
}

// Admission is the call gate for one model name and version.
type Admission struct {
	// Status is 503 while the model is loading and 404 when the name or version was not selected.
	// It is 0 when a run may proceed and when the model is unavailable.
	Status int
	// Reason is empty when a run may proceed. An unavailable model keeps its load reason.
	Reason string
	// Version is the selected version when a run may proceed.
	Version string
}

// Proceed reports whether a run may start.
func (a Admission) Proceed() bool {
	return a.Status == 0 && a.Reason == "" && a.Version != ""
}

// StatusError is a call-gate refusal.
type StatusError struct {
	// Status is the call-gate status.
	Status int
	// Reason explains the refusal.
	Reason string
}

// Error returns the refusal reason.
func (e *StatusError) Error() string {
	if e == nil {
		return ""
	}
	if e.Reason != "" {
		return e.Reason
	}
	return "request was refused"
}

// Model is the load result for one child directory.
type Model struct {
	// Name is the directory name.
	Name string
	// State is ready, loading, or unavailable.
	State State
	// Reason is empty when State is ready.
	Reason string
	// QueueDelayMicroseconds is the queue delay from the model file, or zero when the file omits it.
	QueueDelayMicroseconds uint64
	// Versions are the sessions opened for this model, highest version first.
	Versions []Version
}

// Version is one selected model version.
type Version struct {
	// Version is the decimal version. A directory with no version subdirectories is "1".
	Version string
	// MaxBatchSize is the recorded batch cap. Zero means the model is not batched.
	MaxBatchSize int
	// Batchable reports whether every input and output has a variable leading dimension.
	Batchable bool
	// Warmed reports whether a warm-up run completed.
	Warmed bool
}

// Input is one tensor supplied to a run.
type Input struct {
	// Name is the model input name.
	Name string
	// Shape is the concrete shape. A negative length is rejected.
	Shape []int64
	// Data is the tensor bytes in host order.
	// When Data is aligned for its element type and its capacity holds the tensor, those bytes are the tensor memory.
	// Otherwise the bytes are copied into a pooled buffer for the run and returned afterward.
	Data []byte
}

// Output is one tensor produced by a run.
type Output struct {
	// Name is the model output name.
	Name string
	// Shape is the concrete shape returned by the run.
	Shape []int64
	// Buffer is a copy of the runtime output. Release returns it to the pool.
	Buffer Buffer
}
