// Command inference loads server settings and serves HTTP and gRPC health checks.
// Invalid settings or an unusable Edgelet signing key stop the process before it listens.
package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"google.golang.org/grpc"

	"github.com/eclipse-iofog/ai-inference-onnx/internal/auth"
	"github.com/eclipse-iofog/ai-inference-onnx/internal/config"
	"github.com/eclipse-iofog/ai-inference-onnx/internal/grpcapi"
	"github.com/eclipse-iofog/ai-inference-onnx/internal/httpapi"
)

const (
	// headerTimeout bounds how long a connection may sit before sending headers.
	headerTimeout = 10 * time.Second
	// stopTimeout bounds how long in-flight health checks may finish after a signal.
	stopTimeout = 5 * time.Second
)

func main() {
	os.Exit(run(log.New(os.Stderr, "", log.LstdFlags)))
}

func run(logger *log.Logger) int {
	cfg, err := config.Load()
	if err != nil {
		logger.Printf("configuration: %v", err)
		return 1
	}
	checker, err := auth.New(cfg, "")
	if err != nil {
		logger.Printf("authentication: %v", err)
		return 1
	}

	httpServer := &http.Server{
		Handler:           httpapi.New(checker, serverReady),
		ReadHeaderTimeout: headerTimeout,
		ErrorLog:          logger,
	}
	grpcServer := grpcapi.New(checker, serverReady)

	httpLn, grpcLn, err := listen(cfg)
	if err != nil {
		logger.Printf("listen: %v", err)
		return 1
	}

	errCh := make(chan error, 2)
	go func() {
		errCh <- serveHTTP(httpServer, httpLn)
	}()
	go func() {
		errCh <- serveGRPC(grpcServer, grpcLn)
	}()

	signals := make(chan os.Signal, 1)
	signal.Notify(signals, syscall.SIGINT, syscall.SIGTERM)
	defer signal.Stop(signals)

	select {
	case <-signals:
		if err := stop(httpServer, grpcServer); err != nil {
			logger.Printf("stop: %v", err)
			return 1
		}
		return 0
	case err := <-errCh:
		stopErr := stop(httpServer, grpcServer)
		if err != nil {
			logger.Printf("serve: %v", err)
			return 1
		}
		if stopErr != nil {
			logger.Printf("stop: %v", stopErr)
			return 1
		}
		return 0
	}
}

// serverReady reports whether any loaded model can serve.
// This process has not loaded models, so the set is empty and ready is false.
func serverReady() bool {
	var models []httpapi.State
	return httpapi.AnyReady(models...)
}

func listen(cfg config.Config) (net.Listener, net.Listener, error) {
	httpLn, err := net.Listen("tcp", httpapi.ListenAddr(cfg.HTTPPort))
	if err != nil {
		return nil, nil, fmt.Errorf("http %s: %w", httpapi.ListenAddr(cfg.HTTPPort), err)
	}
	grpcLn, err := net.Listen("tcp", grpcapi.ListenAddr(cfg.GRPCPort))
	if err != nil {
		if closeErr := httpLn.Close(); closeErr != nil {
			return nil, nil, fmt.Errorf("grpc %s: %w", grpcapi.ListenAddr(cfg.GRPCPort), errors.Join(err, closeErr))
		}
		return nil, nil, fmt.Errorf("grpc %s: %w", grpcapi.ListenAddr(cfg.GRPCPort), err)
	}
	return httpLn, grpcLn, nil
}

func serveHTTP(server *http.Server, ln net.Listener) error {
	err := server.Serve(ln)
	if errors.Is(err, http.ErrServerClosed) {
		return nil
	}
	return err
}

func serveGRPC(server *grpcapi.Server, ln net.Listener) error {
	err := server.Serve(ln)
	if errors.Is(err, grpc.ErrServerStopped) {
		return nil
	}
	return err
}

func stop(httpServer *http.Server, grpcServer *grpcapi.Server) error {
	grpcServer.GracefulStop()
	ctx, cancel := context.WithTimeout(context.Background(), stopTimeout)
	defer cancel()
	if err := httpServer.Shutdown(ctx); err != nil {
		return fmt.Errorf("http: %w", err)
	}
	return nil
}
