#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic 90% GPU VRAM Holder
=============================================================================
Maintains GPU memory usage at exactly ~90% of total VRAM on the RTX 5090 (32GB).

Features:
  1. Dynamically calculates target hold:
     Target total memory = 90% of total VRAM (~29,346 MiB).
     Allocates (or deallocates) 128MB chunks to maintain exactly 90% total GPU usage.
  2. Safe auto-yielding (no conflict with training):
     If our training process or any other task needs memory, it automatically shrinks
     its allocation so the active task never suffers Out Of Memory (OOM)!
     When the task finishes and frees memory, it automatically expands back to 90%!
  3. Instant on-demand release ("khi nào cần thì giải phóng"):
     - If /tmp/release_vram or ~/RELEASE_VRAM exists: releases 100% of held memory instantly.
     - Responds to SIGUSR1 (pause/release) and SIGUSR2 (resume).
=============================================================================
"""

import os
import sys
import time
import signal
import logging
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
LOGGER = logging.getLogger("VRAMHolder90")

TARGET_PERCENT = 0.90      # 90% total GPU memory used
CHUNK_SIZE_MB = 128        # Allocate in 128MB chunks for granular control
SAFE_MIN_FREE_MB = 1500    # Always ensure at least 1.5GB of free headroom for active GPU tasks
RELEASE_FILES = [
    Path("/tmp/release_vram"),
    Path.home() / "RELEASE_VRAM",
    Path(__file__).resolve().parent / "RELEASE_VRAM"
]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

paused = False
blocks = []


def handle_pause(signum, frame):
    global paused, blocks
    LOGGER.info("Received pause signal (SIGUSR1). Releasing all held VRAM...")
    paused = True
    blocks.clear()
    torch.cuda.empty_cache()


def handle_resume(signum, frame):
    global paused
    LOGGER.info("Received resume signal (SIGUSR2). Resuming 90% VRAM holding...")
    paused = False


signal.signal(signal.SIGUSR1, handle_pause)
signal.signal(signal.SIGUSR2, handle_resume)


def is_release_requested() -> bool:
    return any(p.exists() for p in RELEASE_FILES)


def main():
    global paused, blocks
    LOGGER.info("======================================================================")
    LOGGER.info("STARTING DYNAMIC 90% VRAM HOLDER DAEMON")
    LOGGER.info(f"Target GPU: {torch.cuda.get_device_name(0)}")
    LOGGER.info(f"Target Utilization: {TARGET_PERCENT*100:.0f}% of total VRAM")
    LOGGER.info("======================================================================")

    chunk_bytes = CHUNK_SIZE_MB * 1024 * 1024
    min_free_bytes = SAFE_MIN_FREE_MB * 1024 * 1024
    last_log_time = 0

    while True:
        try:
            # Check for external release request
            if is_release_requested():
                if blocks:
                    LOGGER.info("Release file detected. Freeing all held VRAM...")
                    blocks.clear()
                    torch.cuda.empty_cache()
                time.sleep(1.0)
                continue

            if paused:
                time.sleep(1.0)
                continue

            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            target_used_bytes = int(total_bytes * TARGET_PERCENT)
            current_held_bytes = len(blocks) * chunk_bytes
            
            # Other processes usage = Total - (Free + Held by us)
            other_used_bytes = total_bytes - (free_bytes + current_held_bytes)
            
            # Ideal amount we should hold to bring total GPU usage to exactly 90%
            target_hold_bytes = max(0, target_used_bytes - other_used_bytes)

            # Safety constraint: never leave less than SAFE_MIN_FREE_MB free
            max_allowable_hold = max(0, current_held_bytes + free_bytes - min_free_bytes)
            final_target_hold = min(target_hold_bytes, max_allowable_hold)

            target_block_count = final_target_hold // chunk_bytes

            if len(blocks) < target_block_count:
                # Allocate more
                needed = target_block_count - len(blocks)
                to_add = min(needed, 8)
                for _ in range(to_add):
                    blk = torch.empty(chunk_bytes, dtype=torch.uint8, device=device)
                    blocks.append(blk)
            elif len(blocks) > target_block_count:
                # Yield memory to active training process
                excess = len(blocks) - target_block_count
                to_remove = min(excess, 16)
                for _ in range(to_remove):
                    blocks.pop()
                torch.cuda.empty_cache()

            # Heartbeat logging every 30 seconds
            now = time.time()
            if now - last_log_time >= 30:
                total_used_mb = (total_bytes - free_bytes) / (1024 * 1024)
                total_mb = total_bytes / (1024 * 1024)
                held_mb = (len(blocks) * chunk_bytes) / (1024 * 1024)
                LOGGER.info(f"GPU VRAM Status: Used {total_used_mb:.0f}/{total_mb:.0f} MiB ({total_used_mb/total_mb*100:.1f}%) | Held by us: {held_mb:.0f} MiB ({len(blocks)} chunks)")
                last_log_time = now

            time.sleep(0.5)

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if blocks:
                    blocks.pop()
                    torch.cuda.empty_cache()
                time.sleep(0.5)
            else:
                LOGGER.error(f"Unexpected RuntimeError: {e}")
                time.sleep(2.0)
        except Exception as e:
            LOGGER.error(f"Error in VRAM holder loop: {e}")
            time.sleep(2.0)


if __name__ == "__main__":
    main()
