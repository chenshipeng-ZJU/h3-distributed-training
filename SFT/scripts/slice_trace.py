#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Slice a huge xprofiler Chrome-Tracing JSON into a browser-loadable subset.

Chrome (V8) cannot parse JSON text larger than ~512MB; injected xprofiler
traces of multi-XPU training easily exceed that (8 cards x 180s ~ 4GB).
This script streams through traceEvents, prints a per-process summary, and
optionally writes a subset filtered by pid / time window.

Usage:
  python3 slice_trace.py SRC.json                      # scan only
  python3 slice_trace.py SRC.json DST.json \
      --pids 17923,"XPU0 HW(0000:61:00.0)" [--t0 1000000] [--t1 2000000]

Pids may be integers (rank processes) or strings (XPU device tracks).
Filter rules when DST is given:
  - metadata events (ph == 'M') are always kept
  - other events are kept iff pid in --pids and (no ts, or t0 <= ts <= t1)

ts/dur follow Chrome Tracing units (microseconds).
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst", nargs="?")
    ap.add_argument("--pids", default="")
    ap.add_argument("--t0", type=float)
    ap.add_argument("--t1", type=float)
    args = ap.parse_args()

    keep_pids = set()
    for p in args.pids.split(","):
        p = p.strip()
        if p:
            keep_pids.add(p)

    with open(args.src, "r", encoding="utf-8", errors="replace") as f:
        s = f.read()

    i = s.find('"traceEvents"')
    if i < 0:
        raise SystemExit("no traceEvents in " + args.src)
    i = s.find("[", i)
    if i < 0:
        raise SystemExit("no '[' after traceEvents")
    i += 1

    dec = json.JSONDecoder()
    n = len(s)

    proc_names = {}
    # pid -> [count, dur_sum, ts_min, ts_max]
    stats = {}
    kept = []

    def handle(o):
        ph = o.get("ph")
        pid = o.get("pid")
        ts = o.get("ts")
        if ph == "M":
            if o.get("name") == "process_name":
                proc_names[pid] = o.get("args", {}).get("name", "?")
            if args.dst is not None:
                kept.append(o)
            return
        if pid is not None:
            st = stats.setdefault(pid, [0, 0.0, None, None])
            st[0] += 1
            d = o.get("dur")
            if d:
                st[1] += d
            if ts is not None:
                if st[2] is None or ts < st[2]:
                    st[2] = ts
                if st[3] is None or ts > st[3]:
                    st[3] = ts
        if args.dst is None:
            return
        if str(pid) not in keep_pids:
            return
        if ts is not None:
            if args.t0 is not None and ts < args.t0:
                return
            if args.t1 is not None and ts > args.t1:
                return
        kept.append(o)

    while i < n:
        while i < n and s[i] in " \t\r\n,":
            i += 1
        if i >= n or s[i] == "]":
            break
        o, i = dec.raw_decode(s, i)
        handle(o)

    print("%-24s %-40s %12s %12s %16s %16s" % (
        "pid", "process", "events", "dur_sum(s)", "ts_min", "ts_max"))
    for pid in sorted(stats, key=lambda p: -stats[p][1]):
        st = stats[pid]
        print("%-24s %-40s %12d %12.1f %16.0f %16.0f" % (
            str(pid)[:24], str(proc_names.get(pid, "?"))[:40],
            st[0], st[1] / 1e6, st[2] or 0.0, st[3] or 0.0))

    if args.dst is not None:
        with open(args.dst, "w") as f:
            json.dump({"traceEvents": kept}, f)
        print("kept %d events -> %s" % (len(kept), args.dst))


if __name__ == "__main__":
    main()
