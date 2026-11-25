import json

with open("/home/nguyendinhquy/TransMTL_OntoKG-main/experiments/fact_to_caption/sft_dataset_verified_train.json", "r") as f:
    dataset = json.load(f)

tr_len = len(dataset["train"])
va_len = len(dataset["val"])
all_items = dataset["train"] + dataset["val"]
print(f"Total items: {len(all_items)} (Train: {tr_len}, Val: {va_len})")

hedging_actions = {"ASSERT": 0, "HEDGE": 0, "OMIT": 0}
total_kg_facts = 0
total_sg_facts = 0

for item in all_items:
    for f in item.get("top5_knowledge_facts", []):
        total_kg_facts += 1
        act = f.get("policy_label", "UNKNOWN")
        if act in hedging_actions:
            hedging_actions[act] += 1
    total_sg_facts += len(item.get("scene_graph_facts", []))

print(f"Total KG facts: {total_kg_facts}")
print(f"Policy label distribution: {hedging_actions}")
print(f"Total Scene Graph facts: {total_sg_facts} (100% ASSERT unhedged)")

for idx, i in enumerate([0, 10, 25]):
    s = all_items[i]
    vid = s["clip_id"]
    lm = s.get("landmark_name", "Unknown")
    print(f"\n=== SAMPLE {idx+1} (clip: {vid}, landmark: {lm}) ===")
    print("Scene Graph (Visual Grounding - 100% ASSERT):")
    for sg in s.get("scene_graph_facts", [])[:3]:
        print(f"  - [{sg['policy_label']}] {sg['fact_text']}")
    print("Top KG Facts (Re-ranked by mu_F, Epistemic certainty c = mu_F * mu_L):")
    for kf in s.get("top5_knowledge_facts", []):
        ft = kf["fact_text"]
        h = kf["policy_label"]
        mf = kf["mu_F"]
        ml = kf["mu_L"]
        c = kf["certainty_c"]
        rev = kf.get("review_result", "N/A")
        print(f"  - [{h}] {ft} | mu_F={mf:.4f}, mu_L={ml:.4f} => c={c:.4f} (gold_review: {rev})")
    cap = s.get("target_caption")
    print(f"Generated Verified Caption (Qwen2.5-14B Teacher):\n  \"{cap}\"")
