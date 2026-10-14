// Package grpcapi serves inference over HTTP/2 without TLS.
// Live and ready stay open. Every other call is checked before it runs.
package grpcapi

import (
	"context"
	"net"
	"strconv"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"

	"github.com/eclipse-iofog/ai-inference-onnx/api/openinference/v2"
	"github.com/eclipse-iofog/ai-inference-onnx/internal/auth"
)

const (
	statusUnauthenticated = "unauthenticated"
	statusForbidden       = "forbidden"
)

// ListenAddr returns the cleartext gRPC listen address for port.
// The host is empty, so the listener accepts every interface.
func ListenAddr(port int) string {
	return net.JoinHostPort("", strconv.Itoa(port))
}

// Server answers live and ready checks.
// Other calls are rejected after the credential check, until a handler exists.
type Server struct {
	openinference.UnimplementedGRPCInferenceServiceServer

	checker *auth.Checker
	ready   func() bool
	rpc     *grpc.Server
}

// New returns a server that uses checker for protected calls.
// ready reports whether any model can serve. A nil ready func means not ready.
// The server speaks HTTP/2 without TLS.
func New(checker *auth.Checker, ready func() bool) *Server {
	s := &Server{checker: checker, ready: ready}
	s.rpc = grpc.NewServer(grpc.UnaryInterceptor(s.intercept))
	openinference.RegisterGRPCInferenceServiceServer(s.rpc, s)
	return s
}

// Serve accepts connections on lis until Stop or GracefulStop.
func (s *Server) Serve(lis net.Listener) error {
	return s.rpc.Serve(lis)
}

// GracefulStop stops new calls and waits for active calls to finish.
func (s *Server) GracefulStop() {
	s.rpc.GracefulStop()
}

// Stop stops the server at once.
func (s *Server) Stop() {
	s.rpc.Stop()
}

// ServerLive reports that this process can answer.
func (s *Server) ServerLive(context.Context, *openinference.ServerLiveRequest) (*openinference.ServerLiveResponse, error) {
	return &openinference.ServerLiveResponse{Live: true}, nil
}

// ServerReady reports whether any model can serve.
// A false result is still a successful response.
func (s *Server) ServerReady(context.Context, *openinference.ServerReadyRequest) (*openinference.ServerReadyResponse, error) {
	return &openinference.ServerReadyResponse{Ready: s.readyNow()}, nil
}

func (s *Server) readyNow() bool {
	if s.ready == nil {
		return false
	}
	return s.ready()
}

func (s *Server) intercept(ctx context.Context, req any, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (any, error) {
	fullMethod := ""
	if info != nil {
		fullMethod = info.FullMethod
	}
	model, version := requestModel(req)
	decision, err := s.checker.AuthorizeGRPC(fullMethod, model, version, authorizationHeader(ctx), time.Now())
	if err != nil || decision != auth.Allow {
		if decision == auth.Forbidden {
			return nil, status.Error(codes.PermissionDenied, statusForbidden)
		}
		return nil, status.Error(codes.Unauthenticated, statusUnauthenticated)
	}
	return handler(ctx, req)
}

func requestModel(req any) (string, string) {
	switch msg := req.(type) {
	case *openinference.ModelInferRequest:
		return msg.GetModelName(), msg.GetModelVersion()
	case *openinference.ModelMetadataRequest:
		return msg.GetName(), msg.GetVersion()
	case *openinference.ModelReadyRequest:
		return msg.GetName(), msg.GetVersion()
	default:
		return "", ""
	}
}

func authorizationHeader(ctx context.Context) string {
	md, ok := metadata.FromIncomingContext(ctx)
	if !ok {
		return ""
	}
	values := md.Get("authorization")
	if len(values) == 0 {
		return ""
	}
	return values[0]
}
