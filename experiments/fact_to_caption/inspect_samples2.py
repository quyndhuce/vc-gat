import json

with open("/home/nguyendinhquy/TransMTL_OntoKG-main/experiments/fact_to_caption/sft_dataset_verified_train.json", "r") as f:
    d = json.load(f)

items = d["train"]
hedged_samples = [x for x in items if any(f["policy_label"] == "HEDGE" for f in x["top5_knowledge_facts"])]
asserted_samples = [x for x in items if any(f["policy_label"] == "ASSERT" for f in x["top5_knowledge_facts"])]

print(f"Hedged samples count: {len(hedged_samples)}")
print(f"Asserted samples count: {len(asserted_samples)}")

if hedged_samples:
    s = hedged_samples[2]
    print("\n=== REPRESENTATIVE HEDGED SAMPLE ===")
    print(f"Clip: {s['clip_id']}, Landmark: {s['landmark_name']}")
    print("Scene Graph:")
    for sg in s["scene_graph_facts"][:2]:
        print(f"  - [{sg['policy_label']}] {sg['fact_text']}")
    print("Top KG Facts:")
    for kf in s["top5_knowledge_facts"]:
        print(f"  - [{kf['policy_label']}] {kf['fact_text']} (c={kf['certainty_c']:.4f})")
    print(f"Caption:\n  \"{s['target_caption']}\"")

if asserted_samples:
    s = asserted_samples[2]
    print("\n=== REPRESENTATIVE ASSERTED SAMPLE ===")
    print(f"Clip: {s['clip_id']}, Landmark: {s['landmark_name']}")
    print("Scene Graph:")
    for sg in s["scene_graph_facts"][:2]:
        print(f"  - [{sg['policy_label']}] {sg['fact_text']}")
    print("Top KG Facts:")
    for kf in s["top5_knowledge_facts"]:
        print(f"  - [{kf['policy_label']}] {kf['fact_text']} (c={kf['certainty_c']:.4f})")
    print(f"Caption:\n  \"{s['target_caption']}\"")
