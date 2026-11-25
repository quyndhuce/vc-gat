"""
OntoKG_EN: End-to-end Demonstration of Module 1, Module 2 & Module 3 for English.
Runs preprocessing, sentence splitting, POS tagging,
Zero-Shot Named Entity Recognition (GLiNER), Conceptual Phrase Extraction,
and International Linked Data (LOD) Entity Linking with Wikidata & MPNet.
"""

import json
import sys
from pathlib import Path

# Add project root to sys.path so OntoKG_EN can be imported directly
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from OntoKG_EN.module1_preprocess_en import Module1PreprocessorEN
from OntoKG_EN.module2_ner_concept_en import Module2NerConceptExtractorEN
from OntoKG_EN.module3_entity_linking_en import Module3EntityLinkerEN


def main():
    print("=" * 85)
    print("OntoKG_EN — English Knowledge Graph Construction: Modules 1, 2 & 3 Demo")
    print("=" * 85)

    # Sample English article
    sample_article = {
        "article_id": "art_en_001",
        "title": "Russian President Vladimir Putin Issues Stark Nuclear Warning to NATO Over Ukraine Conflict",
        "summary": "In a speech in Moscow, President Vladimir Putin warned NATO and the United States about supplying long-range ballistic missile systems to Kyiv.",
        "content": (
            "Russian President Vladimir Putin issued a new warning on October 15, 2024, declaring that "
            "direct military intervention by NATO in Ukraine would trigger unprecedented retaliation. "
            "Speaking to senior defense officials in Moscow, Putin emphasized that Russia's tactical nuclear weapons "
            "and hypersonic missile arsenals are fully operational. Meanwhile, defense ministers from the "
            "United States and the European Union reiterated their commitments to collective defense under the "
            "North Atlantic Treaty. The Kremlin stated that diplomatic channels remain constrained by Western sanctions."
        ),
        "publish_time": "2024-10-15 14:30:00",
        "topic": "International Security",
        "cleaned_keywords": [
            "nuclear deterrence",
            "ballistic missile systems",
            "hypersonic missiles",
            "collective defense",
        ],
    }

    # Seed ontology concepts for Module 2 semantic linking
    seed_ontology_concepts = [
        {"id": "onto:NuclearDeterrence", "name": "nuclear deterrence"},
        {"id": "onto:BallisticMissile", "name": "ballistic missile systems"},
        {"id": "onto:CollectiveDefense", "name": "collective defense"},
        {"id": "onto:Sanctions", "name": "economic sanctions"},
    ]

    # -------------------------------------------------------------
    # 1. RUN MODULE 1: Preprocessing, POS, Tokens, Sentences
    # -------------------------------------------------------------
    print("\n[Step 1] Executing Module 1 (English Preprocessing)...")
    preprocessor = Module1PreprocessorEN(use_pos=True, include_summary_in_full_text=True)
    processed = preprocessor.preprocess_row(sample_article)
    processed_dict = processed.to_dict()

    print(f"✓ Article ID        : {processed_dict['article_id']}")
    print(f"✓ Normalized Title  : {processed_dict['title']}")
    print(f"✓ Topics            : {processed_dict['topic_list']}")
    print(f"✓ Total Sentences   : {len(processed_dict['full_text_sentences'])}")
    print(f"✓ Total Tokens      : {len(processed_dict['full_text_tokens'])}")
    print(f"✓ Total POS Tags    : {len(processed_dict['full_text_pos'])}")
    print("✓ Sample POS Tags   :")
    for tok, pos in processed_dict["full_text_pos"][:8]:
        print(f"    • {tok:<15} ({pos})")

    # -------------------------------------------------------------
    # 2. RUN MODULE 2: GLiNER NER & Concept Extraction
    # -------------------------------------------------------------
    print("\n[Step 2] Executing Module 2 (GLiNER 34 Domain Entities & Concept Extraction)...")
    extractor = Module2NerConceptExtractorEN(
        ontology_concepts=seed_ontology_concepts,
        use_gliner=True,
        gliner_threshold=0.45,
        concept_link_threshold=0.80,
        concept_review_threshold=0.60,
    )
    result = extractor.process_record(processed_dict)

    # -------------------------------------------------------------
    # DISPLAY RESULTS
    # -------------------------------------------------------------
    print("\n" + "=" * 40 + " EXTRACTED NAMED ENTITIES (NER) " + "=" * 40)
    print(f"{'Surface Form':<35} | {'Label':<8} | {'Fine-Grained Type':<24} | {'Conf':<6}")
    print("-" * 85)
    for ent in result["ner_entities"]:
        fg = ent.get("fine_grained_label") or ent.get("label_name", "")
        print(
            f"{ent['surface']:<35} | {ent['label']:<8} | {fg:<24} | {ent['confidence']:<6.2f}"
        )

    print("\n" + "=" * 40 + " EXTRACTED CONCEPTS & LINKING " + "=" * 40)
    print(f"{'Concept Phrase':<35} | {'Score':<8} | {'Action':<10} | {'Linked Target':<25}")
    print("-" * 85)
    for c in result["concept_mentions"]:
        target = c["linked_ontology_id"] or "-"
        print(f"{c['concept']:<35} | {c['score']:<8.2f} | {c['action']:<10} | {target:<25}")

    # -------------------------------------------------------------
    # 3. RUN MODULE 3: Linked Data Entity Linking (Wikidata + MPNet)
    # -------------------------------------------------------------
    print("\n[Step 3] Executing Module 3 (Linked Data Entity Linking with Wikidata & MPNet)...")
    cache_path = current_dir / "wikidata_cache_en.json"
    linker = Module3EntityLinkerEN(
        embedding_model="sentence-transformers/all-mpnet-base-v2",
        wikidata_cache_path=str(cache_path),
        similarity_threshold=0.88,
        use_wikidata=True,
        use_embedding_matching=True,
    )
    linked_result = linker.process_record(result)

    print("\n" + "=" * 38 + " LINKED ENTITIES (W3C LOD / WIKIDATA) " + "=" * 39)
    print(f"{'Surface Form':<28} | {'Label':<6} | {'Source':<14} | {'Wikidata':<10} | {'Linked URI'}")
    print("-" * 105)
    for ent in linked_result["ner_entities"]:
        wd = ent.get("wikidata_id") or "-"
        print(
            f"{ent['surface']:<28} | {ent['label']:<6} | {ent['uri_source']:<14} | {wd:<10} | {ent['uri']}"
        )

    print("\n" + "=" * 40 + " LINKED CONCEPTS & METADATA " + "=" * 41)
    print(f"✓ Article URI : {linked_result.get('article_uri')}")
    print(f"✓ Topic URIs  : {linked_result.get('topic_uris')}")
    for c in linked_result["concept_mentions"]:
        print(f"  • Concept: {c['concept']:<25} -> {c['uri']} ({c['uri_source']})")

    # Save output
    output_path = current_dir / "demo_output_en.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(linked_result, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Saved linked result to: {output_path}")
    print("=" * 85)


if __name__ == "__main__":
    main()

