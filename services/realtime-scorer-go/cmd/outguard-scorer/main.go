package main

import (
	"flag"
	"log"
	"net/http"
	"time"

	scorer "github.com/outguard-med/realtime-scorer-go"
)

func main() {
	bundlePath := flag.String("bundle", "../../artifacts/model_bundle.json", "Path to exported model_bundle.json")
	addr := flag.String("addr", ":8090", "HTTP listen address")
	flag.Parse()

	bundle, err := scorer.LoadModelBundle(*bundlePath)
	if err != nil {
		log.Fatalf("Failed to load OutGuard model bundle: %v", err)
	}
	srv := scorer.NewHTTPServer(bundle)
	httpSrv := &http.Server{
		Addr:              *addr,
		Handler:           srv.Routes(),
		ReadHeaderTimeout: 3 * time.Second,
	}
	log.Printf("[OutGuard-MED Go Scorer] Loaded %d trees & 3 specialists; listening on %s", len(bundle.LGBMTrees), *addr)
	if err := httpSrv.ListenAndServe(); err != nil {
		log.Fatalf("Server exited: %v", err)
	}
}
