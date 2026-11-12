// Package scorer implements ultra-low-latency cross-language inference for OutGuard-MED.
// It evaluates the exact LightGBM tree ensemble, domain-specialist logistic scorers,
// log-odds signal combiner, smooth Isotonic+Platt calibration, and deterministic guardrails
// exported by the Python training pipeline.
package scorer

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
)

// TreeNode represents either an internal split node or a leaf in a LightGBM tree.
type TreeNode struct {
	SplitFeature *int      `json:"split_feature,omitempty"`
	Threshold    float64   `json:"threshold,omitempty"`
	LeftChild    *TreeNode `json:"left_child,omitempty"`
	RightChild   *TreeNode `json:"right_child,omitempty"`
	LeafValue    float64   `json:"leaf_value,omitempty"`
}

func (n *TreeNode) Eval(features []float64) float64 {
	cur := n
	for cur.LeftChild != nil && cur.SplitFeature != nil {
		if features[*cur.SplitFeature] <= cur.Threshold {
			cur = cur.LeftChild
		} else {
			cur = cur.RightChild
		}
	}
	return cur.LeafValue
}

type SpecialistModel struct {
	FeatureIndices []int     `json:"feature_indices"`
	Mean           []float64 `json:"mean"`
	Scale          []float64 `json:"scale"`
	Coef           []float64 `json:"coef"`
	Intercept      float64   `json:"intercept"`
}

func (s *SpecialistModel) PredictProba(features []float64) float64 {
	z := s.Intercept
	for i, fIdx := range s.FeatureIndices {
		scaled := (features[fIdx] - s.Mean[i]) / s.Scale[i]
		z += scaled * s.Coef[i]
	}
	return sigmoid(z)
}

type MetaCombiner struct {
	Coef      []float64 `json:"coef"`
	Intercept float64   `json:"intercept"`
}

type CalibrationModel struct {
	IsoWeight float64   `json:"iso_weight"`
	IsoX      []float64 `json:"iso_x"`
	IsoY      []float64 `json:"iso_y"`
	PlattA    float64   `json:"platt_a"`
	PlattB    float64   `json:"platt_b"`
}

type GuardrailParams struct {
	RecruiterColdOutreachRateMin float64 `json:"recruiter_cold_outreach_rate_min"`
	SelfPersonalEmailNameSimMin  float64 `json:"self_personal_email_name_sim_min"`
	HardBlockPIIPrefixGapMin     float64 `json:"hard_block_pii_prefix_gap_min"`
}

type ThresholdParams struct {
	NudgeThreshold float64 `json:"nudge_threshold"`
	BlockThreshold float64 `json:"block_threshold"`
}

type GoldenVector struct {
	Features          []float64 `json:"features"`
	ExpectedScore     float64   `json:"expected_score"`
	ExpectedLGBMProb  float64   `json:"expected_lgbm_prob"`
	ExpectedAction    string    `json:"expected_action"`
	ExpectedGuardrail string    `json:"expected_guardrail"`
}

type ModelBundle struct {
	Version           string                     `json:"version"`
	FeatureNames      []string                   `json:"feature_names"`
	Specialists       map[string]SpecialistModel `json:"specialists"`
	LGBMTrees         []TreeNode                 `json:"lgbm_trees"`
	MetaCombiner      MetaCombiner               `json:"meta_combiner"`
	Calibration       CalibrationModel           `json:"calibration"`
	Guardrails        GuardrailParams            `json:"guardrails"`
	Thresholds        ThresholdParams            `json:"thresholds"`
	GoldenTestVectors []GoldenVector             `json:"golden_test_vectors"`

	featIndex map[string]int
	graphSpec SpecialistModel
	lexSpec   SpecialistModel
	dlpSpec   SpecialistModel
}

type ScoreResult struct {
	CalibratedRiskScore float64 `json:"calibrated_risk_score"`
	RawLGBMProb         float64 `json:"raw_lgbm_prob"`
	GraphSpecProb       float64 `json:"graph_specialist_prob"`
	LexicalSpecProb     float64 `json:"lexical_specialist_prob"`
	DLPSpecProb         float64 `json:"dlp_specialist_prob"`
	Action              string  `json:"action"`
	TriggeredGuardrail  string  `json:"triggered_guardrail,omitempty"`
}

func LoadModelBundle(path string) (*ModelBundle, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read model bundle: %w", err)
	}
	var b ModelBundle
	if err := json.Unmarshal(raw, &b); err != nil {
		return nil, fmt.Errorf("parse model bundle: %w", err)
	}
	b.featIndex = make(map[string]int, len(b.FeatureNames))
	for i, name := range b.FeatureNames {
		b.featIndex[name] = i
	}
	b.graphSpec = b.Specialists["graph_specialist"]
	b.lexSpec = b.Specialists["lexical_coherence_specialist"]
	b.dlpSpec = b.Specialists["dlp_context_specialist"]
	return &b, nil
}

func (b *ModelBundle) Score(features []float64, allowlistedDomainHit bool) (ScoreResult, error) {
	if len(features) != len(b.FeatureNames) {
		return ScoreResult{}, fmt.Errorf("expected %d features, got %d", len(b.FeatureNames), len(features))
	}

	pGraph := b.graphSpec.PredictProba(features)
	pLex := b.lexSpec.PredictProba(features)
	pDLP := b.dlpSpec.PredictProba(features)

	margin := 0.0
	for i := range b.LGBMTrees {
		margin += b.LGBMTrees[i].Eval(features)
	}
	pLGBM := sigmoid(margin)

	z := b.MetaCombiner.Intercept +
		b.MetaCombiner.Coef[0]*safeLogit(pGraph) +
		b.MetaCombiner.Coef[1]*safeLogit(pLex) +
		b.MetaCombiner.Coef[2]*safeLogit(pDLP) +
		b.MetaCombiner.Coef[3]*safeLogit(pLGBM)

	pIso := interp1D(z, b.Calibration.IsoX, b.Calibration.IsoY)
	pPlatt := sigmoid(b.Calibration.PlattA*z + b.Calibration.PlattB)
	calProb := b.Calibration.IsoWeight*pIso + (1.0-b.Calibration.IsoWeight)*pPlatt

	finalProb, rule := b.applyGuardrails(features, calProb, allowlistedDomainHit)

	action := "ALLOW"
	if finalProb >= b.Thresholds.BlockThreshold {
		action = "HOLD_BLOCK"
	} else if finalProb >= b.Thresholds.NudgeThreshold {
		action = "NUDGE_USER"
	}

	return ScoreResult{
		CalibratedRiskScore: finalProb,
		RawLGBMProb:         pLGBM,
		GraphSpecProb:       pGraph,
		LexicalSpecProb:     pLex,
		DLPSpecProb:         pDLP,
		Action:              action,
		TriggeredGuardrail:  rule,
	}, nil
}

func (b *ModelBundle) applyGuardrails(feat []float64, baseProb float64, allowlisted bool) (float64, string) {
	nudgeTau := b.Thresholds.NudgeThreshold
	blockTau := b.Thresholds.BlockThreshold

	if allowlisted {
		return math.Min(baseProb*0.12, nudgeTau*0.45), "GUARDRAIL_SUBSIDIARY_ALLOWLIST"
	}

	hasSSN := feat[b.featIndex["dlp_has_ssn_or_tax_id"]] > 0.5
	hasSecret := feat[b.featIndex["dlp_has_api_or_cloud_secret"]] > 0.5
	hasCard := feat[b.featIndex["dlp_has_payment_card_luhn"]] > 0.5
	hardDLP := hasSSN || hasSecret || hasCard
	blast := feat[b.featIndex["dlp_composite_blast_radius"]]

	prefixGap := feat[b.featIndex["lex_autocomplete_prefix_collision_gap"]]
	decayAff := feat[b.featIndex["graph_min_recipient_decay_affinity"]]
	lookalike := feat[b.featIndex["lex_domain_is_lookalike_unseen"]] > 0.5
	attMismatch := feat[b.featIndex["sem_attachment_org_mismatch"]] > 0.5
	bodyOtherOrg := feat[b.featIndex["sem_body_mentions_other_known_org"]] > 0.5
	salMismatch := feat[b.featIndex["lex_salutation_mismatch_flag"]] > 0.5
	warmIntro := feat[b.featIndex["lex_has_cc_warm_intro"]] > 0.5
	selfWebmail := feat[b.featIndex["lex_self_personal_webmail_match"]]
	coldRate := feat[b.featIndex["graph_sender_cold_outreach_rate"]]
	verifiedReply := feat[b.featIndex["graph_is_reply_chain_verified"]] > 0.5
	mixedThread := feat[b.featIndex["sem_mixed_internal_external_thread"]] > 0.5
	firstExt := feat[b.featIndex["graph_any_first_time_external"]] > 0.5
	orgDomCnt := feat[b.featIndex["graph_min_org_domain_prior_count"]]

	if hardDLP && firstExt && prefixGap >= b.Guardrails.HardBlockPIIPrefixGapMin && (bodyOtherOrg || attMismatch || salMismatch) {
		floor := math.Max(blockTau+0.04, 0.94)
		return math.Max(baseProb, math.Min(0.995, floor)), "RULE_CRITICAL_PII_PREFIX_COLLISION"
	}
	if lookalike && firstExt && blast >= 0.22 {
		floor := math.Max(nudgeTau+0.05, 0.78)
		if hardDLP {
			floor = math.Max(blockTau+0.02, 0.90)
		}
		return math.Max(baseProb, math.Min(0.99, floor)), "RULE_LOOKALIKE_TYPOSQUAT_SENSITIVE"
	}
	if attMismatch && hardDLP && !warmIntro && (firstExt || bodyOtherOrg) {
		floor := math.Max(blockTau+0.03, 0.92)
		return math.Max(baseProb, math.Min(0.992, floor)), "RULE_WRONG_ORG_ATTACHMENT_PII"
	}

	if selfWebmail >= b.Guardrails.SelfPersonalEmailNameSimMin && !hardDLP && !attMismatch && !bodyOtherOrg {
		return baseProb * 0.12, "GUARDRAIL_SELF_PERSONAL_WEBMAIL"
	}
	if warmIntro && !hardDLP && !attMismatch && !salMismatch {
		return baseProb * 0.18, "GUARDRAIL_CC_WARM_INTRO_EXEMPTION"
	}
	if coldRate >= b.Guardrails.RecruiterColdOutreachRateMin && !hardDLP && !attMismatch && !bodyOtherOrg && !salMismatch && !lookalike {
		return baseProb * 0.10, "GUARDRAIL_HIGH_COLD_OUTREACH_ROLE"
	}
	if verifiedReply && !mixedThread && !attMismatch && !bodyOtherOrg && !salMismatch {
		return baseProb * 0.22, "GUARDRAIL_VERIFIED_REPLY_THREAD"
	}
	if (decayAff >= 0.35 || orgDomCnt >= 3.6) && !mixedThread && !attMismatch && !bodyOtherOrg && !salMismatch && !lookalike {
		return baseProb * 0.22, "GUARDRAIL_ESTABLISHED_PARTNER_COHERENT"
	}

	return baseProb, ""
}

func sigmoid(z float64) float64 {
	if z < -25.0 {
		z = -25.0
	} else if z > 25.0 {
		z = 25.0
	}
	return 1.0 / (1.0 + math.Exp(-z))
}

func safeLogit(p float64) float64 {
	if p < 1e-6 {
		p = 1e-6
	} else if p > 1.0-1e-6 {
		p = 1.0 - 1e-6
	}
	return math.Log(p / (1.0 - p))
}

func interp1D(x float64, xs []float64, ys []float64) float64 {
	n := len(xs)
	if n == 0 {
		return 0.5
	}
	if x <= xs[0] {
		return ys[0]
	}
	if x >= xs[n-1] {
		return ys[n-1]
	}
	idx := sort.SearchFloat64s(xs, x)
	if idx < n && xs[idx] == x {
		return ys[idx]
	}
	x0, x1 := xs[idx-1], xs[idx]
	y0, y1 := ys[idx-1], ys[idx]
	if x1 == x0 {
		return y0
	}
	return y0 + (x-x0)/(x1-x0)*(y1-y0)
}
