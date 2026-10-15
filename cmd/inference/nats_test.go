package main

import (
	"errors"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
	"syscall"
	"testing"
	"time"

	"github.com/nats-io/jwt/v2"
	"github.com/nats-io/nats-server/v2/server"
	"github.com/nats-io/nats.go"
	"github.com/nats-io/nkeys"
)

const natsReplyWait = 5 * time.Second

func TestNATSPartialSettingsDoNotDial(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name string
		nats func(url, creds string) map[string]string
	}{
		{name: "server url only", nats: func(url, _ string) map[string]string {
			return map[string]string{"NATS_SERVER_URL": url}
		}},
		{name: "credentials only", nats: func(_, creds string) map[string]string {
			return map[string]string{"NATS_CREDS_PATH": creds}
		}},
		{name: "empty credentials", nats: func(url, _ string) map[string]string {
			return map[string]string{"NATS_SERVER_URL": url, "NATS_CREDS_PATH": "  "}
		}},
		{name: "empty server url", nats: func(_, creds string) map[string]string {
			return map[string]string{"NATS_SERVER_URL": "", "NATS_CREDS_PATH": creds}
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			addr, accepted := countingListener(t)
			path, _, _ := writeProcessCredentials(t)
			env := tc.nats("nats://"+addr, path)
			env["AUTH_MODE"] = "disabled"
			ports := pickPorts(t)
			proc := startProcess(t, processEnv(t, ports, env))
			t.Cleanup(proc.kill)

			waitLive(t, ports.http)
			assertGET(t, ports.http, "/v2/health/ready", http.StatusBadRequest)
			assertGRPCHealth(t, ports.grpc, false)
			stdout := waitLog(t, proc, "grpc listener is up")
			time.Sleep(300 * time.Millisecond)
			if n := accepted.Load(); n != 0 {
				t.Fatalf("the process dialed %d times", n)
			}
			for _, line := range decodeLog(t, stdout+proc.stdout.String()) {
				if strings.Contains(strings.ToLower(line.Msg), "nats") {
					t.Fatalf("nats log line %+v", line)
				}
			}
		})
	}
}

func TestNATSRefusedDialKeepsServing(t *testing.T) {
	t.Parallel()
	const token = "nats-refused-token-3f8a1c90d2b7e645"
	path, userJWT, seed := writeProcessCredentials(t)
	ports := pickPorts(t)
	proc := startProcess(t, processEnv(t, ports, map[string]string{
		"AUTH_MODE":         "static",
		"AUTH_STATIC_TOKEN": token,
		"NATS_SERVER_URL":   "nats://" + refusedAddress(t),
		"NATS_CREDS_PATH":   path,
	}))
	t.Cleanup(proc.kill)

	waitLive(t, ports.http)
	stdout := waitLog(t, proc, "nats connection attempt failed, retrying")
	assertGET(t, ports.http, "/v2/health/live", http.StatusOK)
	assertGET(t, ports.http, "/v2/health/ready", http.StatusBadRequest)
	assertGRPCHealth(t, ports.grpc, false)
	assertRefused(t, hostPort(ports.metrics))

	var up, retries int
	for _, line := range decodeLog(t, stdout) {
		switch line.Msg {
		case "nats listener is up":
			if line.Level != "INFO" {
				t.Fatalf("listener line %+v", line)
			}
			up++
		case "nats connection attempt failed, retrying":
			if line.Level != "WARN" {
				t.Fatalf("retry line %+v", line)
			}
			retries++
		case "nats connected":
			t.Fatalf("connected to a refused address: %+v", line)
		}
	}
	if up != 1 || retries == 0 {
		t.Fatalf("listener lines %d retry lines %d:\n%s", up, retries, stdout)
	}
	select {
	case <-proc.waited:
		t.Fatalf("process exited\n%s", proc.output())
	default:
	}
	out := proc.output()
	for _, secret := range []string{token, userJWT, seed, "BEGIN NATS"} {
		if strings.Contains(out, secret) {
			t.Fatalf("log contains a secret:\n%s", out)
		}
	}

	if err := proc.signal(syscall.SIGTERM); err != nil {
		t.Fatalf("signal: %v", err)
	}
	if err := proc.waitTimeout(t, 5*time.Second); err != nil {
		t.Fatalf("exit after signal: %v\n%s", err, proc.output())
	}
}

func TestNATSServesEmptyModelSet(t *testing.T) {
	t.Parallel()
	url := startNATSServer(t)
	path, _, _ := writeProcessCredentials(t)
	ports := pickPorts(t)
	proc := startProcess(t, processEnv(t, ports, map[string]string{
		"AUTH_MODE":       "disabled",
		"NATS_SERVER_URL": url,
		"NATS_CREDS_PATH": path,
	}))
	t.Cleanup(proc.kill)
	waitLive(t, ports.http)
	waitLog(t, proc, "nats connected")

	nc, err := nats.Connect(url, nats.NoReconnect())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(nc.Close)
	snapshots, err := nc.SubscribeSync("inference.metrics.>")
	if err != nil {
		t.Fatal(err)
	}

	if status, body := natsRequest(t, nc, "inference.v2.health.live", nil); status != http.StatusOK || len(body) != 0 {
		t.Fatalf("nats live %d %q", status, body)
	}
	if status, body := natsRequest(t, nc, "inference.v2.health.ready", nil); status != http.StatusBadRequest || len(body) != 0 {
		t.Fatalf("nats ready %d %q", status, body)
	}
	status, body := natsRequest(t, nc, "inference.v2.models.add.infer", []byte(`{"inputs":[]}`))
	if status != http.StatusNotFound || string(body) != `{"error":"model \"add\" was not found"}` {
		t.Fatalf("nats infer %d %s", status, body)
	}
	if _, err := nc.Request("inference.v2.repository.models.add.load", nil, time.Second); !errors.Is(err, nats.ErrNoResponders) {
		t.Fatalf("nats load: %v", err)
	}

	assertGET(t, ports.http, "/v2/health/live", http.StatusOK)
	status, body = post(t, ports.http, "/v2/repository/models/add/load", "")
	if status != http.StatusNotImplemented {
		t.Fatalf("load %d %s", status, body)
	}
	assertRefused(t, hostPort(ports.metrics))
	if msg, err := snapshots.NextMsg(300 * time.Millisecond); !errors.Is(err, nats.ErrTimeout) {
		t.Fatalf("a snapshot was published: %v %v", msg, err)
	}
}

// countingListener accepts connections on a local port and counts them.
func countingListener(t *testing.T) (string, *atomic.Int32) {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = ln.Close() })
	var accepted atomic.Int32
	go func() {
		for {
			conn, err := ln.Accept()
			if err != nil {
				return
			}
			accepted.Add(1)
			_ = conn.Close()
		}
	}()
	return ln.Addr().String(), &accepted
}

// refusedAddress is a local address whose listener was opened and closed, so a dial is refused.
func refusedAddress(t *testing.T) string {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr := ln.Addr().String()
	if err := ln.Close(); err != nil {
		t.Fatal(err)
	}
	return addr
}

func startNATSServer(t *testing.T) string {
	t.Helper()
	srv, err := server.NewServer(&server.Options{Host: "127.0.0.1", Port: server.RANDOM_PORT, NoLog: true, NoSigs: true})
	if err != nil {
		t.Fatal(err)
	}
	srv.Start()
	t.Cleanup(srv.Shutdown)
	if !srv.ReadyForConnections(natsReplyWait) {
		t.Fatal("embedded NATS server did not start")
	}
	return srv.ClientURL()
}

// writeProcessCredentials mints a user credentials file in a temp directory and returns its path, JWT, and seed.
func writeProcessCredentials(t *testing.T) (path, userJWT, seed string) {
	t.Helper()
	account, err := nkeys.CreateAccount()
	if err != nil {
		t.Fatal(err)
	}
	user, err := nkeys.CreateUser()
	if err != nil {
		t.Fatal(err)
	}
	userPub, err := user.PublicKey()
	if err != nil {
		t.Fatal(err)
	}
	userJWT, err = jwt.NewUserClaims(userPub).Encode(account)
	if err != nil {
		t.Fatal(err)
	}
	rawSeed, err := user.Seed()
	if err != nil {
		t.Fatal(err)
	}
	contents, err := jwt.FormatUserConfig(userJWT, rawSeed)
	if err != nil {
		t.Fatal(err)
	}
	path = filepath.Join(t.TempDir(), "user.creds")
	if err := os.WriteFile(path, contents, 0o600); err != nil {
		t.Fatal(err)
	}
	return path, userJWT, string(rawSeed)
}

// natsRequest sends one request and joins the reply stream up to its terminator.
// The status is on the first message. The terminator is the only empty message with no headers.
func natsRequest(t *testing.T, nc *nats.Conn, subject string, body []byte) (int, []byte) {
	t.Helper()
	inbox := nc.NewInbox()
	sub, err := nc.SubscribeSync(inbox)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = sub.Unsubscribe() }()
	if err := nc.PublishMsg(&nats.Msg{Subject: subject, Reply: inbox, Data: body}); err != nil {
		t.Fatal(err)
	}
	status := -1
	var joined []byte
	for i := 0; ; i++ {
		msg, err := sub.NextMsg(natsReplyWait)
		if err != nil {
			t.Fatalf("%s reply %d: %v", subject, i, err)
		}
		if i == 0 {
			status, err = strconv.Atoi(msg.Header.Get("Inference-Status"))
			if err != nil {
				t.Fatalf("%s status: %v", subject, err)
			}
		} else if len(msg.Header) == 0 && len(msg.Data) == 0 {
			return status, joined
		}
		joined = append(joined, msg.Data...)
	}
}
