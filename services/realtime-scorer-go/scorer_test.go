package scorer_test

import (
	"bytes"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"

	scorer "github.com/outguard-med/realtime-scorer-go"
)

func loadBundleOrFail(tb testing.TB) *scorer.ModelBundle {
	tb.Helper()
	path := filepath.Join("..", "..", "artifacts", "model_bundle.json")
	b, err := scorer.LoadModelBundle(path)
	if err != nil {
		tb.Fatalf("failed to load model bundle %s: %v", path, err)
	}
	return b
}

func TestPythonCrossLanguageParity(t *testing.T) {
	b := loadBundleOrFail(t)
	if len(b.GoldenTestVectors) == 0 {
		t.Fatal("expected non-empty golden_test_vectors")
	}

	maxScoreDiff := 0.0
	maxLGBMDiff := 0.0

	for idx, gv := range b.GoldenTestVectors {
		res, err := b.Score(gv.Features, false)
		if err != nil {
			t.Fatalf("case %d failed: %v", idx, err)
		}
		lgbmDiff := math.Abs(res.RawLGBMProb - gv.ExpectedLGBMProb)
		scoreDiff := math.Abs(res.CalibratedRiskScore - gv.ExpectedScore)
		if lgbmDiff > maxLGBMDiff {
			maxLGBMDiff = lgbmDiff
		}
		if scoreDiff > maxScoreDiff {
			maxScoreDiff = scoreDiff
		}
		if lgbmDiff > 2e-5 {
			t.Errorf("case %d LGBM prob mismatch: got %.7f want %.7f (diff %.2e)", idx, res.RawLGBMProb, gv.ExpectedLGBMProb, lgbmDiff)
		}
		if scoreDiff > 2e-5 {
			t.Errorf("case %d calibrated score mismatch: got %.7f want %.7f (diff %.2e)", idx, res.CalibratedRiskScore, gv.ExpectedScore, scoreDiff)
		}
		if res.Action != gv.ExpectedAction {
			t.Errorf("case %d policy action mismatch: got %s want %s", idx, res.Action, gv.ExpectedAction)
		}
		if res.TriggeredGuardrail != gv.ExpectedGuardrail {
			t.Errorf("case %d guardrail mismatch: got %q want %q", idx, res.TriggeredGuardrail, gv.ExpectedGuardrail)
		}
	}
	t.Logf("Verified %d Python<->Go golden vectors | max LGBM diff = %.2e | max H4 score diff = %.2e",
		len(b.GoldenTestVectors), maxLGBMDiff, maxScoreDiff)
}

func TestHTTPScoreAndAllowlistOverride(t *testing.T) {
	b := loadBundleOrFail(t)
	srv := scorer.NewHTTPServer(b)
	handler := srv.Routes()

	gv := b.GoldenTestVectors[0]
	payload, _ := json.Marshal(scorer.FeatureScoreRequest{
		EventID:              "evt_go_test_01",
		Features:             gv.Features,
		AllowlistedDomainHit: true,
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/score-features", bytes.NewReader(payload))
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200 OK, got %d: %s", rec.Code, rec.Body.String())
	}
	var resp scorer.FeatureScoreResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if resp.TriggeredGuardrail != "GUARDRAIL_SUBSIDIARY_ALLOWLIST" || resp.Action != "ALLOW" {
		t.Fatalf("expected subsidiary allowlist override -> ALLOW, got action=%s rule=%s", resp.Action, resp.TriggeredGuardrail)
	}
}

func BenchmarkRealtimeEnsembleScore(b *testing.B) {
	bundle := loadBundleOrFail(b)
	vec := bundle.GoldenTestVectors[0].Features
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, err := bundle.Score(vec, false)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkRealtimeEnsembleScoreParallel(b *testing.B) {
	bundle := loadBundleOrFail(b)
	vec := bundle.GoldenTestVectors[0].Features
	b.ResetTimer()
	b.ReportAllocs()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			_, err := bundle.Score(vec, false)
			if err != nil {
				b.Fatal(err)
			}
		}
	})
}
