"""
Epistemic Prompt Construction under Assert/Hedge/Omit Regimes (Section 4.3).
"""

from typing import List, Dict, Any


def format_fact_packet(
    admitted_facts: List[Dict[str, Any]],
    tau_h: float = 0.80,
    tau_d: float = 0.50
) -> str:
    """
    Formats admitted facts into epistemic instruction prompt:
      c >= tau_h: ASSERT (direct assertion)
      tau_d <= c < tau_h: HEDGE (epistemic modality, e.g., 'historically associated with')
      c < tau_d: OMIT (suppress assertion)
    """
    lines = []
    lines.append("You are an expert cultural heritage tour guide. Describe the video clip accurately.")
    lines.append("Verbalize the following verified evidence packet according to specified epistemic regimes:")

    assert_facts = []
    hedge_facts = []

    for f in admitted_facts:
        triple = f"{f['subject']} -- {f['predicate']} -> {f['object']}"
        conf = float(f.get("confidence", 0.0))
        if conf >= tau_h:
            assert_facts.append(triple)
        elif conf >= tau_d:
            hedge_facts.append(triple)

    if assert_facts:
        lines.append("\n[Direct Assertions (High Confidence)]:")
        for t in assert_facts:
            lines.append(f"  - Assert definitively: {t}")

    if hedge_facts:
        lines.append("\n[Hedged Claims (Moderate Confidence - Use Hedged Lexicon)]:")
        for t in hedge_facts:
            lines.append(f"  - Attribute conditionally: {t} (e.g., 'believed to be', 'associated with')")

    lines.append("\nStrict Requirement: Do NOT assert ungrounded details or unhedged speculative claims.")
    return "\n".join(lines)
