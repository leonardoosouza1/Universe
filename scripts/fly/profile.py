#!/usr/bin/env python3
"""Sample Linux process CPU/RSS and global DRM GPU counters without device readback."""
import argparse
import csv
import os
from pathlib import Path
import time


def integer(path):
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def children(parent):
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[1]) == parent:
                yield int(entry.name), fields
        except (OSError, ValueError, IndexError):
            continue


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--device", type=Path, default=Path("/sys/class/drm/card1/device"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    previous = {}
    hz = os.sysconf("SC_CLK_TCK")
    pagesize = os.sysconf("SC_PAGE_SIZE")
    with args.output.open("x", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["unix_s", "pid", "cpu_percent_one_core", "rss_bytes", "gpu_busy_percent_global",
                         "memory_busy_percent_global", "vram_bytes_global", "gtt_bytes_global"])
        while Path(f"/proc/{args.parent_pid}").exists():
            now = time.monotonic()
            for pid, fields in children(args.parent_pid):
                cpu_seconds = (int(fields[11]) + int(fields[12])) / hz
                old = previous.get(pid)
                cpu = 100 * (cpu_seconds - old[1]) / (now - old[0]) if old else None
                previous[pid] = (now, cpu_seconds)
                writer.writerow([time.time(), pid, cpu, int(fields[21]) * pagesize,
                                 integer(args.device / "gpu_busy_percent"), integer(args.device / "mem_busy_percent"),
                                 integer(args.device / "mem_info_vram_used"), integer(args.device / "mem_info_gtt_used")])
            file.flush()
            time.sleep(1)


if __name__ == "__main__":
    main()
