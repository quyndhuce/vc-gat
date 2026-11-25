#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/hcmc_02_build_hcmc_knowledge_and_fkg.py
=============================================================================
Stage 2 for Unseen HCMC Generalization Experiment (PRE-TEST, ZERO VIDEO LEAKAGE):
1. Collects multi-source encyclopedic knowledge for 5 unseen HCMC landmarks +
   Ho Chi Minh City macro node:
     - Source 1 (Wikipedia + Wikidata API, confidence=95)
     - Source 2 (Curated Web Search & Official Heritage Portals, confidence=75)
     - Source 3 (LLM Parametric Knowledge: Qwen2.5-14B-Instruct, confidence=60)
     - Source 4 (LLM Parametric Knowledge: Llama-2-7b-chat-hf, confidence=60)
     - Source 5 (Architectural Visual Description derived from Wiki+Web, conf=80)
2. Builds pre-test Visual Descriptive Prompts per HCMC landmark (same style as
   experiments/config.py LANDMARK_DESCRIPTIVE_PROMPTS) strictly from text.
3. Extends SharedNodeRegistry & extracts per-KB FKGs via GLiNER-Relex v1.0,
   Meronym Anchoring, Landmark-Centric Filtering, and Module 5 Dempster/T-conorm.
4. Merges the 6 new HCMC KBs with the 233 existing Hanoi KBs into an isolated
   global FKG directory `data/fkg_per_kb_with_hcmc/` (without mutating the
   original Hanoi `data/fkg_per_kb/`).
=============================================================================
"""

import gc
import json
import logging
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
LOGGER = logging.getLogger("HCMC_Knowledge_FKG")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "OntoKG_EN"))

from OntoKG_EN.build_fkg_per_kb import (
    DocumentGraphExtractor,
    SharedNodeRegistry,
    clean_wikipedia_document,
    export_document_ttl,
    filter_landmark_centric_triples,
)
from OntoKG_EN.module5_fkg_builder_en import FKGBuilder
from OntoKG_EN.build_global_merged_fkg import fuse_global_fkg

DATA_DIR = REPO_ROOT / "data"
ORIG_KB_EXPORT = DATA_DIR / "sources" / "kb_full_export_2026-10-01.json"
ORIG_ALIASES = REPO_ROOT / "OntoKG_EN" / "aliases_en.json"
ORIG_FKG_DIR = DATA_DIR / "fkg_per_kb"

OUT_BENCH_DIR = DATA_DIR / "hcmc_benchmark"
EXT_FKG_DIR = DATA_DIR / "fkg_per_kb_with_hcmc"

LOCAL_QWEN_14B = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-14B-Instruct")
LOCAL_LLAMA_7B = os.path.expanduser("~/.cache/huggingface/hub/models--NousResearch--Llama-2-7b-chat-hf")

# Curated web search summaries from official tourism & heritage portals
WEB_SEARCH_TEXTS: Dict[str, str] = {
    "ho_chi_minh_city": (
        "Ho Chi Minh City, historically known as Saigon, is the largest city in Vietnam and the economic, "
        "cultural, and heritage center of southern Vietnam. Situated along the Saigon River, the city preserves "
        "an extraordinary concentration of French colonial architecture in District 1 alongside historic Chinese-Vietnamese "
        "assembly halls and temples in Cholon (District 5), as well as iconic 20th-century modernist monuments."
    ),
    "saigon_central_post_office": (
        "Source: Ho Chi Minh City Tourism & Heritage Architectural Guides\n"
        "The Saigon Central Post Office (Buu dien Trung tam Sai Gon) stands on Cong Xa Paris Square in District 1, "
        "directly across from Notre-Dame Cathedral Basilica of Saigon. Following an earlier 1860s Saigon Wire Department "
        "building, construction of the current grand post office headquarters began in 1886 and was inaugurated in 1891. "
        "Designed by French architects Auguste Henri Villedieu and Marie-Alfred Foulhoux (often popularly misattributed to "
        "Gustave Eiffel due to its exposed metal trusses), the building blends Gothic, Renaissance, and French colonial "
        "architectural styles with subtle Eastern decorative details. The symmetrical yellow exterior features arched windows, "
        "white ornamental pilasters, decorative relief plaques inscribed with inventors of electricity and telegraphy, and a "
        "large Parisian circular facade clock above the central entrance. Inside, the soaring barrel-vaulted ceiling is "
        "supported by green-painted iron arches and wrought-iron pillars reminiscent of a nineteenth-century European railway "
        "station. Two large historical maps hang on the interior entrance walls: 'Lignes telegraphiques du Sud Vietnam et du "
        "Cambodge (1892)' and 'Saigon et ses environs (1892)'. The main hall preserves antique wooden telephone booths, "
        "classic wooden service counters, patterned tiled floors, and a prominent portrait of President Ho Chi Minh at the far end."
    ),
    "independence_palace": (
        "Source: Independence Palace Official Relic Portal (dinhdoclap.gov.vn) & Heritage Records\n"
        "Independence Palace (Dinh Doc Lap), also officially named Reunification Hall (Hoi truong Thong Nhat), is a landmark "
        "historical monument located on Nam Ky Khoi Nghia Street in District 1, Ho Chi Minh City, surrounded by a vast shaded "
        "park of ancient tropical trees and expansive front lawns with a circular fountain. Its precursor was Norodom Palace, "
        "built by French colonial authorities between 1868 and 1871 as the residence of the Governor of Cochinchina. After a "
        "1962 aerial bombing severely damaged the left wing, the old colonial structure was demolished and a new modernist palace "
        "was designed by Vietnamese architect Ngo Viet Thu, the first Asian laureate of the Grand Prix de Rome. Construction began "
        "in July 1962 and the building was inaugurated on October 31, 1966. Ngo Viet Thu integrated 1960s tropical modernist architecture "
        "with traditional East Asian philosophy: the ground plan resembles the character Cat symbolizing good fortune, vertical concrete "
        "sun-shading louvers across the white facade evoke stylized bamboo stalks, and the rooftop pavilion combined with the central "
        "flagpole forms the character Trung symbolizing loyalty. Inside are state banquet and presidential reception halls with lacquer "
        "paintings, underground war command rooms with vintage maps and telecommunications equipment, and a rooftop helipad where a historic "
        "UH-1 helicopter sits on display. Historic tanks number 843 and 390 crashed through the wrought-iron entrance gates on April 30, 1975, "
        "marking the end of the Vietnam War and national reunification."
    ),
    "ben_thanh_market": (
        "Source: Ben Thanh Market Heritage & Saigon Urban Architectural Archives\n"
        "Ben Thanh Market (Cho Ben Thanh) in District 1, Ho Chi Minh City, is one of the oldest and most recognizable commercial "
        "landmarks in Saigon. Its name originates from the seventeenth century, when informal markets gathered near the Ben Nghe River "
        "wharf beside the Gia Dinh Citadel. After draining the Marais Boresse swamp in 1911, French contractor Etablissements Brossard et Mopin "
        "constructed the new market hall between 1912 and March 1914 over 13,000 square meters, originally naming it Les Halles Centrales de Saigon. "
        "Its signature architectural feature is the prominent cream-yellow southern clock tower with three clock faces overlooking Quach Thi Trang "
        "Square. Four wide main entrance gates—South, North, East, and West—open onto busy District 1 streets, adorned above with distinctive "
        "terracotta ceramic bas-reliefs depicting agricultural produce, cattle, and fish. Designed in an Indochinese-French colonial style adapted "
        "for tropical ventilation, the spacious interior hall features high corrugated metal and skylight trusses, wide walkways, and tightly packed "
        "stalls selling textiles, ao dai fabrics, conical hats, handicrafts, dried goods, coffee, spices, fresh produce, and traditional southern "
        "Vietnamese street food."
    ),
    "saigon_opera_house": (
        "Source: Saigon Opera House Architectural Records & Vietnam Tourism Heritage\n"
        "The Saigon Opera House, officially the Municipal Theatre of Ho Chi Minh City (Nha hat Thanh pho Ho Chi Minh), is located on "
        "Lam Son Square along Dong Khoi Street in District 1. Following a 1895 architectural competition, construction took place from 1898 "
        "and the theatre opened on January 1, 1900. French architects Felix Olivier, Ernest Guichard, and Eugene Ferret designed the monument "
        "in the ornate Flamboyant Beaux-Arts French Third Republic style, inspired by the Petit Palais in Paris. Notable engineering details "
        "include a raised floor roughly two meters above street level and a two-layer soundproof door system to insulate the auditorium from "
        "city noise. Its ivory-white stone facade displays a grand arched central portal framed by Corinthian pilasters, sculpted relief "
        "balustrades, caryatid statues supporting the triangular pediment, winged female angel figures, and ornate metal roof crests "
        "originally crafted in France. After serving as the seat of the National Assembly of South Vietnam from 1956 to 1967, the building "
        "returned to performing arts in 1975, underwent major architectural restoration between 1995 and 1998, and was designated a National "
        "Architectural and Artistic Relic in 2012. Today it hosts classical concerts, ballet, opera, and contemporary Vietnamese bamboo "
        "circus productions such as A O Show."
    ),
    "ba_thien_hau_temple": (
        "Source: Cholon Heritage Archives & Vietnam National Cultural Site Records\n"
        "Ba Thien Hau Temple (Chua Ba Thien Hau), officially the Tue Thanh Assembly Hall (Hoi quan Tue Thanh), stands at 710 Nguyen Trai "
        "Street in the historic Chinatown of Cholon, District 5, Ho Chi Minh City. Founded around 1760 by Cantonese immigrants from Guangzhou "
        "in Guangdong, China, and renovated in 1800, 1842, and 1890, it is dedicated to Mazu (Thien Hau Thanh Mau), the Chinese sea goddess "
        "deified from Lin Moniang who protects sailors, fishermen, and merchants. Built with materials imported from China in a traditional "
        "Quoc-character rectangular layout, the temple connects a front vestibule, an open-air central courtyard (sky well) admitting daylight "
        "and ventilation, and a richly decorated main sanctuary. Its curved Yin-Yang tiled roofs and upper walls are famous for intricate, "
        "colorful Shiwan porcelain and ceramic reliefs depicting scenes from Chinese mythology, dragons, phoenixes, lions, and classical legends. "
        "Inside the dimly lit, atmospheric sanctuary hang hundreds of large red spiral incense coils suspended from the wooden rafters with "
        "pink prayer papers attached, drifting fragrant smoke above carved wooden altars, gilded lacquer calligraphy panels, bronze bells, stone "
        "steles, and the wooden statue of Lady Thien Hau flanked by guardians Thien Ly Nhan and Thuan Phong Nhi. In 1993, it was recognized "
        "as a National Architectural and Artistic Heritage Site."
    ),
}

HCMC_LANDMARKS_META = [
    {
        "kb_id": "ho_chi_minh_city",
        "name": "Ho Chi Minh City",
        "name_vi": "Thành phố Hồ Chí Minh",
        "wiki_title": "Ho Chi Minh City",
        "wiki_vi_title": "Thành phố Hồ Chí Minh",
        "type": "location",
        "parent_kb": None,
        "aliases": ["Ho Chi Minh City", "Saigon", "Sài Gòn", "Thành phố Hồ Chí Minh", "TP.HCM", "HCMC"],
    },
    {
        "kb_id": "saigon_central_post_office",
        "name": "Saigon Central Post Office",
        "name_vi": "Bưu điện trung tâm Sài Gòn",
        "wiki_title": "Saigon Central Post Office",
        "wiki_vi_title": "Bưu điện trung tâm Sài Gòn",
        "type": "object",
        "parent_kb": "ho_chi_minh_city",
        "aliases": [
            "Saigon Central Post Office",
            "Ho Chi Minh City Central Post Office",
            "Bưu điện trung tâm Sài Gòn",
            "Bưu điện Thành phố Hồ Chí Minh",
            "Central Post Office",
        ],
    },
    {
        "kb_id": "independence_palace",
        "name": "Independence Palace",
        "name_vi": "Dinh Độc Lập",
        "wiki_title": "Independence Palace",
        "wiki_vi_title": "Dinh Độc Lập",
        "type": "object",
        "parent_kb": "ho_chi_minh_city",
        "aliases": [
            "Independence Palace",
            "Reunification Palace",
            "Reunification Hall",
            "Norodom Palace",
            "Dinh Độc Lập",
            "Hội trường Thống Nhất",
        ],
    },
    {
        "kb_id": "ben_thanh_market",
        "name": "Ben Thanh Market",
        "name_vi": "Chợ Bến Thành",
        "wiki_title": "Ben Thanh Market",
        "wiki_vi_title": "Chợ Bến Thành",
        "type": "object",
        "parent_kb": "ho_chi_minh_city",
        "aliases": [
            "Ben Thanh Market",
            "Bến Thành Market",
            "Chợ Bến Thành",
            "Les Halles Centrales de Saigon",
        ],
    },
    {
        "kb_id": "saigon_opera_house",
        "name": "Saigon Opera House",
        "name_vi": "Nhà hát Thành phố Hồ Chí Minh",
        "wiki_title": "Municipal Theatre of Ho Chi Minh City",
        "wiki_vi_title": "Nhà hát Thành phố Hồ Chí Minh",
        "type": "object",
        "parent_kb": "ho_chi_minh_city",
        "aliases": [
            "Saigon Opera House",
            "Municipal Theatre of Ho Chi Minh City",
            "Ho Chi Minh City Opera House",
            "Saigon Municipal Opera House",
            "Nhà hát Thành phố Hồ Chí Minh",
            "Nhà hát Lớn Sài Gòn",
        ],
    },
    {
        "kb_id": "ba_thien_hau_temple",
        "name": "Ba Thien Hau Temple",
        "name_vi": "Chùa Bà Thiên Hậu",
        "wiki_title": "Thien Hau Temple of Cholon",
        "wiki_vi_title": "Hội quán Tuệ Thành",
        "type": "object",
        "parent_kb": "ho_chi_minh_city",
        "aliases": [
            "Ba Thien Hau Temple",
            "Thien Hau Temple",
            "Thien Hau Temple of Cholon",
            "Thien Hau Pagoda",
            "Tue Thanh Assembly Hall",
            "Chùa Bà Thiên Hậu",
            "Miếu Thiên Hậu",
            "Hội quán Tuệ Thành",
        ],
    },
]


def resolve_local_hf(cache_dir: str, fallback: str) -> str:
    if os.path.isdir(cache_dir):
        snaps = sorted(Path(cache_dir).glob("snapshots/*/"))
        if snaps:
            return str(snaps[0])
    return fallback


WIKI_HEADERS = {"User-Agent": "TransMTL-HeritageResearch/1.0 (academic@transmtl.org)"}


def _urlopen_with_retry(url: str, retries: int = 5) -> Dict[str, Any]:
    for attempt in range(retries):
        try:
            time.sleep(1.2 + attempt * 1.5)
            req = urllib.request.Request(url, headers=WIKI_HEADERS)
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt + 1 < retries:
                wait = 3.0 * (attempt + 1)
                LOGGER.warning("HTTP %d for %s, backing off %.1fs...", e.code, url[:60], wait)
                time.sleep(wait)
                continue
            raise
    return {}


def fetch_wikipedia_extract(title: str, lang: str = "en") -> Tuple[str, str, Optional[str]]:
    url = (
        f"https://{lang}.wikipedia.org/w/api.php?action=query&prop=extracts|pageprops|info"
        f"&inprop=url&explaintext=1&redirects=1&titles={urllib.parse.quote(title)}&format=json"
    )
    data = _urlopen_with_retry(url)
    pages = data.get("query", {}).get("pages", {})
    for pid, pdata in pages.items():
        if pid != "-1":
            ext = pdata.get("extract", "").strip()
            full_url = pdata.get("fullurl", f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}")
            qid = pdata.get("pageprops", {}).get("wikibase_item")
            return ext, full_url, qid
    return "", "", None


def fetch_wikidata_summary_sentences(qid: Optional[str], landmark_name: str) -> str:
    if not qid:
        return ""
    try:
        api_url = (
            f"https://www.wikidata.org/w/api.php?action=wbgetentities&ids={qid}"
            "&props=descriptions|claims|aliases&languages=en&format=json"
        )
        data = _urlopen_with_retry(api_url)
        ent = data.get("entities", {}).get(qid, {})
        desc = ent.get("descriptions", {}).get("en", {}).get("value", "")
        claims = ent.get("claims", {})
        lines = []
        if desc:
            lines.append(f"{landmark_name} is registered on Wikidata ({qid}) as {desc}.")
        # Extract P571 (inception), P84 (architect) Q-IDs if present
        q_to_resolve = []
        prop_map = {"P84": "architect", "P131": "located_in", "P149": "architectural_style", "P17": "country"}
        claim_pairs = []
        for pcode, plabel in prop_map.items():
            for cl in claims.get(pcode, [])[:2]:
                val = cl.get("mainsnak", {}).get("datavalue", {}).get("value")
                if isinstance(val, dict) and "id" in val:
                    q_to_resolve.append(val["id"])
                    claim_pairs.append((plabel, val["id"]))
        if "P571" in claims:
            for cl in claims["P571"][:1]:
                tval = cl.get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("time", "")
                m = re.search(r"(\d{4})", tval)
                if m:
                    lines.append(f"{landmark_name} was constructed or established in {m.group(1)}.")
        if q_to_resolve:
            ids_str = "|".join(sorted(set(q_to_resolve)))
            lurl = f"https://www.wikidata.org/w/api.php?action=wbgetentities&ids={ids_str}&props=labels&languages=en&format=json"
            ldata = _urlopen_with_retry(lurl)
            labels_map = {
                k: v.get("labels", {}).get("en", {}).get("value", k)
                for k, v in ldata.get("entities", {}).items()
            }
            for plabel, target_q in claim_pairs:
                lbl = labels_map.get(target_q, target_q)
                if plabel == "architect":
                    lines.append(f"{landmark_name} was designed by {lbl}.")
                elif plabel == "architectural_style":
                    lines.append(f"{landmark_name} features {lbl} architecture.")
                elif plabel == "located_in":
                    lines.append(f"{landmark_name} is located in {lbl}.")
                elif plabel == "country":
                    lines.append(f"{landmark_name} is situated in {lbl}.")
        return " ".join(lines)
    except Exception as e:
        LOGGER.warning("Wikidata resolve warning for %s (%s): %s", landmark_name, qid, e)
        return ""


def generate_llm_articles_and_prompts(landmarks_data: List[Dict[str, Any]]) -> None:
    """Uses cached Qwen2.5-14B-Instruct and Llama-2-7b-chat-hf to synthesize
    encyclopedic LLM articles, visual descriptions, and pre-test zero-shot prompts."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)

    # 1. Qwen2.5-14B-Instruct
    qwen_path = resolve_local_hf(LOCAL_QWEN_14B, "Qwen/Qwen2.5-14B-Instruct")
    LOGGER.info("Loading Qwen2.5-14B-Instruct from %s...", qwen_path)
    q_tok = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)
    if q_tok.pad_token is None:
        q_tok.pad_token = q_tok.eos_token
    q_mod = AutoModelForCausalLM.from_pretrained(qwen_path, quantization_config=bnb, device_map={"": device}, trust_remote_code=True)
    q_mod.eval()

    def ask_qwen(sys_txt: str, usr_txt: str, max_tokens: int = 450) -> str:
        msgs = [{"role": "system", "content": sys_txt}, {"role": "user", "content": usr_txt}]
        prompt = q_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = q_tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
        with torch.no_grad():
            out = q_mod.generate(**enc, max_new_tokens=max_tokens, do_sample=False, pad_token_id=q_tok.pad_token_id)
        return q_tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    for lm in landmarks_data:
        name = lm["name"]
        name_vi = lm["name_vi"]
        wiki_text = lm["wiki_extract"][:3000]
        web_text = lm["web_text"][:2500]

        # Parametric encyclopedic article (like LLMQwen in Hanoi)
        llm_qwen_doc = ask_qwen(
            "You are a knowledgeable Vietnamese cultural heritage encyclopedia writer. Write factual, clear English prose with well-formed sentences suitable for knowledge graph extraction.",
            f"Write an informative 3-paragraph encyclopedic article in English about the landmark '{name}' ({name_vi}) in Ho Chi Minh City, Vietnam. Cover its history, founding year, architects/builders, architectural style, physical components, materials, and cultural significance.",
            max_tokens=450,
        )
        lm["llm_qwen_doc"] = llm_qwen_doc

        # Visual description derived from Wiki + Web texts
        vis_desc = ask_qwen(
            "You extract concrete visible architectural and visual descriptions strictly from the provided source text. Write simple, clear sentences stating visual components, colors, materials, shapes, facades, interior structures, and surroundings.",
            f"Based ONLY on the sources below, write 6-8 clear factual sentences describing the visible architectural features, exterior facade, interior hall/courtyard elements, materials, colors, and surroundings of '{name}' ({name_vi}). Use '{name}' as the subject.\n\nSOURCES:\n{wiki_text[:1800]}\n\n{web_text[:1800]}",
            max_tokens=320,
        )
        lm["visual_desc"] = vis_desc

        # 5 Calibrated Descriptive Zero-Shot Prompts (matching config.py format)
        if lm["kb_id"] != "ho_chi_minh_city":
            prompts_raw = ask_qwen(
                "You write concise visual camera-view prompts for image/video zero-shot classification with CLIP/SigLIP. Return ONLY a JSON array of 5 strings.",
                f"Write 5 distinct visual camera-view prompts in English starting with 'a photo of' that uniquely identify '{name}' ({name_vi}) in Ho Chi Minh City, capturing its iconic exterior facade, interior hall/courtyard, distinctive architectural elements, colors, and landmarks.\nReference text:\n{vis_desc}\n\nOutput strictly a JSON list of 5 strings.",
                max_tokens=260,
            )
            try:
                m = re.search(r"\[.*\]", prompts_raw, re.S)
                parsed = json.loads(m.group(0)) if m else []
                parsed = [str(x).strip() for x in parsed if isinstance(x, str) and len(x.strip()) > 10][:5]
            except Exception:
                parsed = []
            if len(parsed) < 3:
                parsed = [s.strip() for s in vis_desc.split(".") if len(s.strip()) > 15][:4]
            lm["descriptive_prompts"] = parsed
        else:
            lm["descriptive_prompts"] = []

    del q_mod, q_tok
    gc.collect()
    torch.cuda.empty_cache()

    # 2. Llama-2-7b-chat-hf (matching LLM LLAMA in Hanoi)
    llama_path = resolve_local_hf(LOCAL_LLAMA_7B, "NousResearch/Llama-2-7b-chat-hf")
    LOGGER.info("Loading Llama-2-7b-chat-hf from %s...", llama_path)
    l_tok = AutoTokenizer.from_pretrained(llama_path, trust_remote_code=True)
    if l_tok.pad_token is None:
        l_tok.pad_token = l_tok.eos_token
    l_mod = AutoModelForCausalLM.from_pretrained(llama_path, quantization_config=bnb, device_map={"": device}, trust_remote_code=True)
    l_mod.eval()

    for lm in landmarks_data:
        name = lm["name"]
        name_vi = lm["name_vi"]
        prompt = (
            f"<s>[INST] <<SYS>>\nYou are an encyclopedia writer specializing in Vietnamese cultural heritage.\n<</SYS>>\n\n"
            f"Write an accurate, factual 3-paragraph overview of '{name}' ({name_vi}) in Ho Chi Minh City, Vietnam, "
            f"covering its location, history, architects or founders, architectural features, materials, and cultural role. [/INST]"
        )
        enc = l_tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            out = l_mod.generate(**enc, max_new_tokens=380, do_sample=False, pad_token_id=l_tok.pad_token_id)
        lm["llm_llama_doc"] = l_tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    del l_mod, l_tok
    gc.collect()
    torch.cuda.empty_cache()


def main():
    OUT_BENCH_DIR.mkdir(parents=True, exist_ok=True)
    EXT_FKG_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # 1. Fetch Wikipedia + Wikidata + attach Web Search corpus
    LOGGER.info("[1/4] Fetching Wikipedia & Wikidata knowledge for HCMC landmarks...")
    landmarks_data = []
    for meta in HCMC_LANDMARKS_META:
        ext, url, qid = fetch_wikipedia_extract(meta["wiki_title"], lang="en")
        wd_summary = fetch_wikidata_summary_sentences(qid, meta["name"])
        full_wiki = f"Source URL: {url}\nArticle Title: {meta['wiki_title']}\n\n{ext}"
        if wd_summary:
            full_wiki += f"\n\n{wd_summary}"
        web_text = WEB_SEARCH_TEXTS.get(meta["kb_id"], "")
        landmarks_data.append({
            **meta,
            "wiki_url": url,
            "wikidata_id": qid,
            "wiki_extract": ext,
            "wiki_doc": full_wiki,
            "web_text": web_text,
        })
        LOGGER.info("  * %-30s | QID=%-9s | Wiki=%5d chars | Web=%5d chars",
                    meta["name"], str(qid), len(full_wiki), len(web_text))

    # 2. Generate LLM parametric articles + visual descriptions + zero-shot prompts
    LOGGER.info("[2/4] Generating LLM parametric articles (Qwen2.5-14B & LLaMA-2-7B) + Visual Descriptions...")
    generate_llm_articles_and_prompts(landmarks_data)

    # Save curated knowledge & prompt bank BEFORE touching any test video
    prompts_export = {
        lm["name"]: lm["descriptive_prompts"]
        for lm in landmarks_data if lm["kb_id"] != "ho_chi_minh_city"
    }
    knowledge_summary_path = OUT_BENCH_DIR / "hcmc_landmarks_knowledge_and_prompts.json"
    with open(knowledge_summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "landmarks": landmarks_data,
            "descriptive_prompts": prompts_export,
        }, f, ensure_ascii=False, indent=2)
    LOGGER.info("Saved HCMC knowledge & pre-test zero-shot prompts -> %s", knowledge_summary_path)

    # 3. Merge into extended KB export + aliases (keeping original Hanoi intact)
    LOGGER.info("[3/4] Extending KB Export & Aliases (233 Hanoi + 6 HCMC nodes)...")
    with open(ORIG_KB_EXPORT, "r", encoding="utf-8") as f:
        orig_kb = json.load(f)
    with open(ORIG_ALIASES, "r", encoding="utf-8") as f:
        aliases_map = json.load(f)

    ext_aliases = dict(aliases_map)
    for lm in landmarks_data:
        ext_aliases[lm["name"]] = lm["aliases"]
        ext_aliases[lm["name_vi"]] = lm["aliases"]
    ext_aliases_path = OUT_BENCH_DIR / "aliases_with_hcmc.json"
    with open(ext_aliases_path, "w", encoding="utf-8") as f:
        json.dump(ext_aliases, f, ensure_ascii=False, indent=2)

    hcmc_oid_base = "69ffhcmc000000000000000"
    new_nodes = []
    hcmc_macro_oid = f"{hcmc_oid_base}0"
    for i, lm in enumerate(landmarks_data):
        oid = hcmc_macro_oid if lm["kb_id"] == "ho_chi_minh_city" else f"{hcmc_oid_base}{i}"
        pid = None if lm["parent_kb"] is None else hcmc_macro_oid
        docs = [
            {
                "doc_id": f"{lm['kb_id']}_wiki",
                "kb_id": lm["kb_id"],
                "source": "Wikipedia",
                "confidence_score": 95,
                "title": f"Wikipedia - {lm['name']}",
                "content": lm["wiki_doc"],
                "url": lm["wiki_url"],
                "tags": ["wikipedia", "crawled_wiki"],
            },
            {
                "doc_id": f"{lm['kb_id']}_web",
                "kb_id": lm["kb_id"],
                "source": "Website",
                "confidence_score": 75,
                "title": f"Web Heritage Archive - {lm['name']}",
                "content": lm["web_text"],
                "url": None,
                "tags": ["website", "web_search"],
            },
            {
                "doc_id": f"{lm['kb_id']}_qwen",
                "kb_id": lm["kb_id"],
                "source": "LLMQwen",
                "confidence_score": 60,
                "title": f"LLM Qwen - {lm['name']}",
                "content": lm["llm_qwen_doc"],
                "url": None,
                "tags": ["llm_extracted", "llmqwen"],
            },
            {
                "doc_id": f"{lm['kb_id']}_llama",
                "kb_id": lm["kb_id"],
                "source": "LLM LLAMA",
                "confidence_score": 60,
                "title": f"LLM LLaMA - {lm['name']}",
                "content": lm["llm_llama_doc"],
                "url": None,
                "tags": ["llm_extracted", "llm_llama"],
            },
            {
                "doc_id": f"{lm['kb_id']}_visual",
                "kb_id": lm["kb_id"],
                "source": "VisualDescription",
                "confidence_score": 80,
                "title": f"Visual Architecture - {lm['name']}",
                "content": lm["visual_desc"],
                "url": None,
                "tags": ["visual_description"],
            },
        ]
        new_nodes.append({
            "_id": oid,
            "id": oid,
            "kb_id": lm["kb_id"],
            "name": lm["name"],
            "name_vi": lm["name_vi"],
            "type": lm["type"],
            "parent_id": pid,
            "aliases": lm["aliases"],
            "wikipedia_url": lm["wiki_url"],
            "documents": docs,
        })

    # Copy old nodes + append new nodes
    orig_kbid_set = {n.get("kb_id") for n in orig_kb.get("nodes_flat", [])}
    merged_nodes = list(orig_kb.get("nodes_flat", []))
    for nn in new_nodes:
        if nn["kb_id"] not in orig_kbid_set:
            merged_nodes.append(nn)

    ext_kb_export_path = OUT_BENCH_DIR / "kb_full_export_with_hcmc.json"
    with open(ext_kb_export_path, "w", encoding="utf-8") as f:
        json.dump({
            "export_info": {**orig_kb.get("export_info", {}), "hcmc_extended": True, "total_nodes": len(merged_nodes)},
            "nodes_flat": merged_nodes,
        }, f, ensure_ascii=False, indent=2)

    # 4. Copy existing 233 Hanoi KB outputs cleanly + run GLiNER & Module 5 on 6 new HCMC KBs + global merge
    LOGGER.info("[4/4] Building per-KB FKGs for HCMC & merging into Global FKG...")
    orig_kb_index_path = ORIG_FKG_DIR / "kb_index.json"
    with open(orig_kb_index_path, "r", encoding="utf-8") as f:
        orig_index = json.load(f)

    # Copy Hanoi KB folders cleanly (only if not yet copied)
    for kb_id in orig_index.get("kbs", {}):
        src_dir = ORIG_FKG_DIR / kb_id
        dst_dir = EXT_FKG_DIR / kb_id
        if src_dir.exists() and not dst_dir.exists():
            shutil.copytree(src_dir, dst_dir)

    cache_path = str(REPO_ROOT / "OntoKG_EN" / "wikidata_cache_en.json")
    registry = SharedNodeRegistry(cache_path=cache_path)
    for lm in landmarks_data:
        if lm.get("wiki_url") and lm.get("wikidata_id"):
            registry.wikidata_cache[lm["wiki_url"]] = lm["wikidata_id"]
    registry.build_from_kb_export(str(ext_kb_export_path), str(ext_aliases_path), allow_network=False)

    registry_json = EXT_FKG_DIR / "shared_node_registry.json"
    registry_pkl = EXT_FKG_DIR / "shared_node_registry.pkl"
    registry.save(registry_json, registry_pkl)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = DocumentGraphExtractor(device=device, threshold=0.25)

    ext_kbs_dict = dict(orig_index.get("kbs", {}))
    new_docs_count = 0
    new_triples_count = 0

    for nn in new_nodes:
        kb_id = nn["kb_id"]
        name = nn["name"]
        docs = nn["documents"]

        kb_folder = EXT_FKG_DIR / kb_id
        kb_docs_folder = kb_folder / "docs"
        kb_docs_folder.mkdir(parents=True, exist_ok=True)
        kb_triples = []
        kb_doc_meta = []

        for d_idx, doc in enumerate(docs, 1):
            doc_id = doc.get("doc_id", f"{kb_id}_doc{d_idx}")
            clean_text, _ = clean_wikipedia_document(doc.get("content", ""))
            (kb_docs_folder / f"{doc_id}_clean.txt").write_text(clean_text, encoding="utf-8")

            raw_triples = extractor.extract_document_triples(
                cleaned_text=clean_text,
                article_id=doc_id,
                kb_id=kb_id,
                registry=registry,
            )
            # Tag authority based on document source confidence
            src_name = doc.get("source", "unknown")
            auth_val = 0.95 if src_name == "Wikipedia" else (0.85 if src_name == "VisualDescription" else (0.80 if src_name == "Website" else 0.70))
            for t in raw_triples:
                t["authority"] = auth_val
                t["source"] = src_name

            triples = filter_landmark_centric_triples(raw_triples, kb_id=kb_id, registry=registry)
            (kb_docs_folder / f"{doc_id}_triples.json").write_text(json.dumps(triples, ensure_ascii=False, indent=2), encoding="utf-8")
            export_document_ttl(triples, doc_id, kb_id, kb_docs_folder / f"{doc_id}_graph.ttl")

            kb_triples.extend(triples)
            new_docs_count += 1
            new_triples_count += len(triples)
            kb_doc_meta.append({"doc_id": doc_id, "source": src_name, "url": doc.get("url"), "n_triples": len(triples)})

        (kb_folder / "kb_meta.json").write_text(
            json.dumps({
                "kb_id": kb_id, "name": name, "name_vi": nn.get("name_vi", ""),
                "type": nn.get("type", "object"), "n_docs": len(docs),
                "n_triples": len(kb_triples), "documents": kb_doc_meta,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if kb_triples:
            fkg_builder = FKGBuilder(conflict_K_threshold=0.50, copy_sim_threshold=0.85)
            triple_files = [str(p) for p in kb_docs_folder.glob("*_triples.json")]
            fkg_builder.build(
                triple_files=triple_files,
                linked_files=[],
                registry_path=str(registry_pkl),
                output_dir=str(kb_folder),
            )
            LOGGER.info("  ✓ HCMC KB %-28s: %3d triples -> %2d SV, %3d MV facts",
                        kb_id, len(kb_triples), len(fkg_builder.fact_groups), len(fkg_builder.fuzzy_facts))

        kb_fkg_facts_path = kb_folder / "fkg_facts.json"
        ext_kbs_dict[kb_id] = {
            "name": name,
            "name_vi": nn.get("name_vi", ""),
            "n_docs": len(docs),
            "n_triples": len(kb_triples),
            "kb_folder": str(kb_folder),
            "fkg_facts": str(kb_fkg_facts_path) if kb_fkg_facts_path.exists() else None,
        }

    ext_index_path = EXT_FKG_DIR / "kb_index.json"
    ext_index_path.write_text(json.dumps({
        "export_info": {"hcmc_extended": True},
        "total_kbs": len(ext_kbs_dict),
        "kbs": ext_kbs_dict,
        "total_new_hcmc_docs": new_docs_count,
        "total_new_hcmc_triples": new_triples_count,
        "elapsed_seconds": round(time.time() - t_start, 2),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    LOGGER.info("Merging all %d KBs (233 Hanoi + 6 HCMC) via build_global_merged_fkg.fuse_global_fkg...", len(ext_kbs_dict))
    summary = fuse_global_fkg(base_dir=EXT_FKG_DIR, output_dir=EXT_FKG_DIR)
    LOGGER.info("GLOBAL EXTENDED FKG COMPLETE in %.1fs: %s", time.time() - t_start, summary)


if __name__ == "__main__":
    main()
