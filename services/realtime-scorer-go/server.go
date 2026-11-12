package scorer

import (
	"encoding/json"
	"net/http"
	"sync/atomic"
	"time"
)

type FeatureScoreRequest struct {
	EventID              string    `json:"event_id"`
	Features             []float64 `json:"features"`
	AllowlistedDomainHit bool      `json:"allowlisted_domain_hit"`
}

type FeatureScoreResponse struct {
	EventID string `json:"event_id"`
	ScoreResult
	LatencyMicros int64 `json:"latency_micros"`
}

type HTTPServer struct {
	bundle       *ModelBundle
	requestCount atomic.Uint64
	totalMicros  atomic.Uint64
}

func NewHTTPServer(bundle *ModelBundle) *HTTPServer {
	return &HTTPServer{bundle: bundle}
}

func (s *HTTPServer) Routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", s.handleHealth)
	mux.HandleFunc("POST /v1/score-features", s.handleScore)
	return mux
}

func (s *HTTPServer) handleHealth(w http.ResponseWriter, _ *http.Request) {
	reqs := s.requestCount.Load()
	meanMicros := 0.0
	if reqs > 0 {
		meanMicros = float64(s.totalMicros.Load()) / float64(reqs)
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"status":              "ok",
		"engine":              "go-native-outguard-med",
		"bundle_version":      s.bundle.Version,
		"num_lgbm_trees":      len(s.bundle.LGBMTrees),
		"scored_requests":     reqs,
		"mean_latency_micros": meanMicros,
	})
}

func (s *HTTPServer) handleScore(w http.ResponseWriter, r *http.Request) {
	t0 := time.Now()
	var req FeatureScoreRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid JSON payload", http.StatusBadRequest)
		return
	}
	res, err := s.bundle.Score(req.Features, req.AllowlistedDomainHit)
	if err != nil {
		http.Error(w, err.Error(), http.StatusUnprocessableEntity)
		return
	}
	micros := time.Since(t0).Microseconds()
	s.requestCount.Add(1)
	s.totalMicros.Add(uint64(micros))

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(FeatureScoreResponse{
		EventID:       req.EventID,
		ScoreResult:   res,
		LatencyMicros: micros,
	})
}
