#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/upload_to_figshare.py
=============================================================================
Automated multi-part dataset uploader for Figshare REST API v2.
Uploads the benchmark dataset, knowledge graphs, and scene-conditioned labels.

Usage:
    export FIGSHARE_TOKEN="your_personal_token_here"
    python3 scripts/upload_to_figshare.py --file /path/to/vc_gat_figshare_dataset.tar.gz

Or interactive:
    python3 scripts/upload_to_figshare.py --token <token> --file <archive_path>
=============================================================================
"""

import os
import sys
import json
import hashlib
import argparse
import requests
from pathlib import Path

API_BASE = "https://api.figshare.com/v2"

ARTICLE_METADATA = {
    "title": "Decision-Certified VC-GAT: Cultural Heritage Video Captioning & Fuzzy Knowledge Graph Benchmark",
    "description": (
        "Official benchmark dataset and knowledge graphs for the paper: "
        "'Decision-Certified Partial Reasoning for Uncertainty-Aware Heritage Video Captioning' "
        "(Submitted to International Journal of Approximate Reasoning, Elsevier).\n\n"
        "This dataset contains:\n"
        "1. Complete fused fuzzy knowledge graphs for 4,093 cultural heritage landmarks.\n"
        "2. Ground-truth Horn complete gold facts and hard-negative contrastive challenge pairs.\n"
        "3. Canonical Scheme A parent-video disjoint benchmark splits (Hanoi-500 In-Domain, "
        "HCMC-300 Zero-Shot Transfer, and Val-149 Calibration dev set).\n"
        "4. Pre-extracted video scene graphs and visual embeddings for direct reproduction without raw video decoding.\n\n"
        "Anonymous Code Repository: https://anonymous.4open.science/r/vc-gat-E1C7/\n"
        "Google Drive Mirror (Data & 51 GB Video Clips): https://drive.google.com/drive/folders/1iTEHWfA5XfYW5Y1KvoFMofbhSpki2Dwv?usp=sharing"
    ),
    "defined_type": "dataset",
    "tags": [
        "cultural heritage",
        "video captioning",
        "fuzzy knowledge graph",
        "horn reasoning",
        "approximate reasoning",
        "conformal prediction",
        "multimodal reasoning"
    ],
    "categories_by_source_id": ["310101"], # Computer Science
    "license": 1 # CC BY 4.0
}


def get_file_md5(filepath: str, chunk_size: int = 1048576) -> str:
    """Calculates MD5 checksum of a file."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def create_article(headers: dict, metadata: dict) -> int:
    """Creates a new article entry on Figshare."""
    print("Creating new article draft on Figshare...")
    url = f"{API_BASE}/account/articles"
    resp = requests.post(url, headers=headers, json=metadata)
    if resp.status_code != 201:
        print(f"Error creating article: HTTP {resp.status_code}")
        print(resp.text)
        sys.exit(1)
    
    article_id = resp.json()["location"].split("/")[-1]
    print(f"✓ Created Figshare Article ID: {article_id}")
    return int(article_id)


def initiate_upload(headers: dict, article_id: int, filepath: str) -> dict:
    """Initiates multi-part file upload with Figshare."""
    file_size = os.path.getsize(filepath)
    file_name = os.path.basename(filepath)
    file_md5 = get_file_md5(filepath)
    
    print(f"Initiating upload for: {file_name} ({file_size / (1024**2):.2f} MB, MD5: {file_md5})...")
    url = f"{API_BASE}/account/articles/{article_id}/files"
    data = {"name": file_name, "size": file_size, "md5": file_md5}
    
    resp = requests.post(url, headers=headers, json=data)
    if resp.status_code != 201:
        print(f"Error initiating upload: HTTP {resp.status_code}")
        print(resp.text)
        sys.exit(1)
    
    file_info_url = resp.json()["location"]
    file_resp = requests.get(file_info_url, headers=headers)
    return file_resp.json()


def upload_parts(headers: dict, file_info: dict, filepath: str):
    """Uploads file parts directly to Figshare S3 endpoint."""
    upload_url = file_info["upload_url"]
    upload_token_resp = requests.get(upload_url, headers=headers)
    parts_data = upload_token_resp.json()
    
    parts = parts_data.get("parts", [])
    print(f"Uploading {len(parts)} parts...")
    
    with open(filepath, "rb") as f:
        for idx, part in enumerate(parts):
            part_num = part["partNo"]
            start_offset = part["startOffset"]
            end_offset = part["endOffset"]
            chunk_size = end_offset - start_offset + 1
            
            f.seek(start_offset)
            chunk_data = f.read(chunk_size)
            
            part_url = f"{upload_url}/{part_num}"
            print(f"  --> Uploading Part {part_num}/{len(parts)} ({chunk_size / (1024**2):.2f} MB)...", end="", flush=True)
            put_resp = requests.put(part_url, data=chunk_data)
            if put_resp.status_code in [200, 201]:
                print(" OK")
            else:
                print(f" FAILED (HTTP {put_resp.status_code})")
                sys.exit(1)


def complete_upload(headers: dict, article_id: int, file_id: int):
    """Signals to Figshare that all parts are uploaded."""
    print("Completing upload on Figshare...")
    url = f"{API_BASE}/account/articles/{article_id}/files/{file_id}"
    resp = requests.post(url, headers=headers)
    if resp.status_code in [200, 202]:
        print("✓ File upload successfully completed and verified by Figshare!")
    else:
        print(f"Warning during upload completion: HTTP {resp.status_code} - {resp.text}")


def main():
    parser = argparse.ArgumentParser(description="Upload dataset archive to Figshare")
    parser.add_argument("--file", type=str, required=True, help="Path to archive file (.tar.gz or .zip)")
    parser.add_argument("--token", type=str, default=None, help="Figshare Personal Access Token")
    parser.add_argument("--publish", action="store_true", help="Publish immediately (default: draft)")
    args = parser.parse_args()

    token = args.token or os.environ.get("FIGSHARE_TOKEN")
    if not token:
        print("\n[!] Error: Figshare API Token is required.")
        print("    Please provide it via --token <your_token> or set FIGSHARE_TOKEN in your environment.")
        print("    To generate a token:")
        print("    1. Log in to https://figshare.com")
        print("    2. Go to Profile -> Applications -> Personal Tokens -> Create Personal Token")
        sys.exit(1)

    filepath = os.path.abspath(args.file)
    if not os.path.exists(filepath):
        print(f"[!] Error: File not found: {filepath}")
        sys.exit(1)

    headers = {"Authorization": f"token {token}", "Content-Type": "application/json"}

    # Test authentication
    auth_resp = requests.get(f"{API_BASE}/account", headers=headers)
    if auth_resp.status_code != 200:
        print(f"[!] Authentication failed: HTTP {auth_resp.status_code}. Please check your FIGSHARE_TOKEN.")
        sys.exit(1)
    
    user_info = auth_resp.json()
    print(f"✓ Authenticated with Figshare as: {user_info.get('first_name', '')} {user_info.get('last_name', '')} ({user_info.get('email', '')})")

    # Step 1: Create article draft
    article_id = create_article(headers, ARTICLE_METADATA)

    # Step 2: Initiate file upload
    file_info = initiate_upload(headers, article_id, filepath)
    file_id = file_info["id"]

    # Step 3: Upload chunks
    upload_parts(headers, file_info, filepath)

    # Step 4: Complete upload
    complete_upload(headers, article_id, file_id)

    print("\n" + "="*80)
    print(f"🎉 DATASET SUCCESSFULLY UPLOADED TO FIGSHARE!")
    print(f"Article Draft URL: https://figshare.com/account/articles/{article_id}")
    print("You can now review your item on the Figshare Web interface, reserve a DOI, or publish it.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
