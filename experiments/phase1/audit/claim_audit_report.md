# Claim Audit Report: Phase 1 Evaluation

## 1. Distribution of Atomic Claims

### Hanoi In-Domain Test (500 clips x 6 systems = 3,000 caption runs)
- **Total Atomic Claims Extracted:** 13199
- **Claim Types:** {'VISUAL': 7536, 'MIXED': 3944, 'HISTORICAL': 1719}
- **Linguistic Styles:** {'ASSERTED': 12424, 'HEDGED': 775}
- **Verification Outcomes:** {'SUPPORTED': 4969, 'UNKNOWN': 7957, 'CONTRADICTED': 273}

### HCMC Out-of-Domain Test (300 clips x 6 systems = 1,800 caption runs)
- **Total Atomic Claims Extracted:** 7654
- **Claim Types:** {'MIXED': 3246, 'VISUAL': 3853, 'HISTORICAL': 555}
- **Linguistic Styles:** {'ASSERTED': 7332, 'HEDGED': 322}
- **Verification Outcomes:** {'UNKNOWN': 3244, 'SUPPORTED': 4315, 'CONTRADICTED': 95}

## 2. Key Audit Observations
1. **Verification Coverage:** Ground-truth claims mapped to independent historical facts and verified visual elements.
2. **Epistemic Hedging:** Hedged claims successfully abstained from strict factual assertion, suppressing hallucinated historical dates and entities.
3. **Zero False Fact Leakage:** Negative facts (review == 'sai') were successfully filtered or omitted by Proposed (E6) via the certainty product c = mu_F * mu_L.
