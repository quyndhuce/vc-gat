import sys
sys.path.append('/home/nguyendinhquy/OvSGTR')
from fkg_fact_extractor import FKGFactExtractor, TEST_CASES

extractor = FKGFactExtractor(threshold=0.42, relation_threshold=0.05, tau_min=0.05, gamma=0.4)
with open('/tmp/cases_output.md', 'w', encoding='utf-8') as fh:
    for case in TEST_CASES:
        extractor.facts.clear()
        extractor.extract(case['text'], source_id=case['id'], subject_hint=case['subject_hint'])
        extractor.resolve_conflicts(0.15)
        data = extractor.export(tau_F=0.2)
        fh.write(f"### {case['name']}\n")
        fh.write("**Văn bản nguồn:**\n")
        fh.write(f"> \"{case['text']}\"\n\n")
        fh.write(f"**Đồ thị tri thức trích xuất ({len(data['edges'])} facts, {len(data['nodes'])} nodes):**\n\n")
        fh.write("| Head | Quan hệ | Tail | Score (Gốc) | Score (Giãn mờ) | $\\mu_F$ (Fuzzy) | Trạng thái |\n")
        fh.write("| :--- | :--- | :--- | :---: | :---: | :---: | :---: |\n")
        for e in data['edges']:
            flag = "`[Caption OK]`" if e['caption_allowed'] else "`[Candidate]`"
            fh.write(f"| **{e['head']}** | `--[{e['relation']}]-->` | `{e['tail']}` | {e['score']:.4f} | {e['diluted_score']:.4f} | {e['mu_F']:.4f} | {flag} |\n")
        fh.write("\n---\n\n")
print("Saved to /tmp/cases_output.md successfully!")
