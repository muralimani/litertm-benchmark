#!/usr/bin/env python3
"""Benchmark LiteRT-LM (Gemma) on a Raspberry Pi.

Measures, while the model is loaded and generating:
  * model load time and memory cost (cold + warm reloads)
  * time-to-first-token, prefill tok/s, decode tok/s (engine-reported and wall-clock)
  * CPU temperature, CPU frequency, and the Pi firmware throttle flags
  * process/system CPU utilisation, RSS, available memory, swap

A background thread samples the sensors every --sample-interval seconds and tags each
sample with the current phase (idle, load1, run p128 #2, sustained, cooldown ...), so
temperature/frequency/memory can be correlated with what the model was doing.

Setup:
    python3 -m venv venv && . venv/bin/activate
    pip install litert-lm-api psutil
    # model, e.g.:
    #   hf download litert-community/gemma-4-E2B-it-litert-lm gemma-4-E2B-it.litertlm

Examples:
    python bench_litert.py --model gemma-4-E2B-it.litertlm
    python bench_litert.py --model m.litertlm --threads 4 --prompt-tokens 32,256,1024
    sudo -E venv/bin/python bench_litert.py --model m.litertlm --drop-caches   # true cold load
    python bench_litert.py --model m.litertlm --sustained-minutes 10          # thermal soak

Results are written to --out-dir: summary.json, runs.csv, samples.csv.
"""

import argparse
import csv
import datetime as dt
import gc
import glob
import json
import os
import platform
import re
import resource
import shutil
import statistics
import subprocess
import sys
import threading
import time
from importlib import metadata

try:
    import psutil
except ImportError:
    sys.exit("psutil is required:  pip install psutil")

MB = 1024 * 1024
UNSTABLE_CV = 0.15  # warn when decode tok/s stdev/mean exceeds this

# Bits of `vcgencmd get_throttled` (Raspberry Pi firmware).
THROTTLE_BITS = {
    0: "under-voltage",
    1: "arm-freq-capped",
    2: "throttled",
    3: "soft-temp-limit",
    16: "under-voltage-occurred",
    17: "freq-cap-occurred",
    18: "throttling-occurred",
    19: "soft-temp-limit-occurred",
}

_FILLER_WORDS = (
    "river mountain quantum lantern violet market engine harbor meadow copper signal "
    "window orchard thunder marble compass velvet anchor garden puzzle ribbon summit "
    "lagoon ember falcon"
).split()
_PROMPT_HEAD = (
    "Write a long, detailed essay about the history of computing. "
    "Use these notes for inspiration:\n"
)


# --------------------------------------------------------------------------- sensors


def _read_text(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def _find_thermal_zone():
    zones = sorted(glob.glob("/sys/class/thermal/thermal_zone*"))
    for zone in zones:
        ztype = (_read_text(f"{zone}/type") or "").lower()
        if any(k in ztype for k in ("cpu", "soc", "pkg")):
            return f"{zone}/temp"
    return f"{zones[0]}/temp" if zones else None


def decode_throttled(value):
    if value is None:
        return []
    return [name for bit, name in THROTTLE_BITS.items() if value & (1 << bit)]


class Sensors:
    """Reads what the platform offers; every reader returns None when unavailable."""

    def __init__(self):
        self.thermal_path = _find_thermal_zone()
        self.vcgencmd = shutil.which("vcgencmd")
        self.freq_path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"
        if _read_text(self.freq_path) is None:
            self.freq_path = None
        self._vcgencmd_broken = False

    def _vcgencmd_run(self, *args):
        if not self.vcgencmd or self._vcgencmd_broken:
            return None
        try:
            out = subprocess.run(
                [self.vcgencmd, *args], capture_output=True, text=True, timeout=2
            )
            return out.stdout.strip() if out.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            self._vcgencmd_broken = True
            return None

    def temp_c(self):
        # A reading <= 0 C means a virtual/unsupported sensor (e.g. a VM's thermal_zone0
        # reports a constant 0), not a real temperature, so it counts as unavailable.
        if self.thermal_path:
            raw = _read_text(self.thermal_path)
            if raw and raw.lstrip("-").isdigit() and int(raw) > 0:
                return int(raw) / 1000.0
        out = self._vcgencmd_run("measure_temp")  # temp=54.0'C
        if out:
            m = re.search(r"([\d.]+)", out)
            if m and float(m.group(1)) > 0:
                return float(m.group(1))
        try:
            for entries in psutil.sensors_temperatures().values():
                if entries and entries[0].current > 0:
                    return entries[0].current
        except (AttributeError, OSError):
            pass
        return None

    def freq_mhz(self):
        if self.freq_path:
            raw = _read_text(self.freq_path)
            if raw and raw.isdigit():
                return int(raw) / 1000.0
        try:
            freq = psutil.cpu_freq()
            return freq.current if freq else None
        except OSError:
            return None

    def throttled(self):
        out = self._vcgencmd_run("get_throttled")  # throttled=0x50005
        if out:
            m = re.search(r"0x([0-9a-fA-F]+)", out)
            if m:
                return int(m.group(1), 16)
        return None

    def describe(self):
        return {
            "temperature": self.thermal_path or ("vcgencmd" if self.vcgencmd else "psutil/none"),
            "frequency": self.freq_path or "psutil/none",
            "throttle_flags": "vcgencmd" if self.vcgencmd else "unavailable (not a Pi?)",
        }


class Sampler(threading.Thread):
    """Samples sensors + resource use on a fixed interval, tagged with the current phase."""

    def __init__(self, sensors, interval):
        super().__init__(daemon=True)
        self.sensors = sensors
        self.interval = interval
        self.phase = "init"
        self.samples = []
        self._halt = threading.Event()  # not `_stop`: that shadows Thread._stop
        self._t0 = time.perf_counter()
        self._proc = psutil.Process()
        self._proc.cpu_percent(None)  # prime the interval counters
        psutil.cpu_percent(None)

    def sample_once(self):
        vm = psutil.virtual_memory()
        try:
            rss = self._proc.memory_info().rss / MB
        except psutil.Error:
            rss = None
        return {
            "t_s": round(time.perf_counter() - self._t0, 2),
            "phase": self.phase,
            "temp_c": self.sensors.temp_c(),
            "freq_mhz": self.sensors.freq_mhz(),
            "sys_cpu_pct": psutil.cpu_percent(None),
            "proc_cpu_pct": self._proc.cpu_percent(None),  # 100 == one full core
            "rss_mb": rss,
            "sys_used_mb": (vm.total - vm.available) / MB,
            "sys_avail_mb": vm.available / MB,
            "swap_used_mb": psutil.swap_memory().used / MB,
            "throttled": self.sensors.throttled(),
        }

    def run(self):
        while not self._halt.is_set():
            self.samples.append(self.sample_once())
            self._halt.wait(self.interval)

    def stop(self):
        self._halt.set()
        self.join(timeout=5)

    def latest(self):
        return self.samples[-1] if self.samples else {}


# --------------------------------------------------------------------------- stats


def percentile(values, p):
    s = sorted(values)
    if not s:
        return None
    k = (len(s) - 1) * p / 100
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return {
        "n": len(vals),
        "mean": statistics.fmean(vals),
        "stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "p50": percentile(vals, 50),
        "max": max(vals),
    }


def fmt(stat, key="mean", digits=2, width=8):
    if not stat or stat.get(key) is None:
        return "n/a".rjust(width)
    return f"{stat[key]:.{digits}f}".rjust(width)


def cv_pct(stat):
    """Coefficient of variation (stdev/mean) as a string, or 'n/a'."""
    if not stat or not stat["mean"] or stat["n"] < 2:
        return "n/a"
    return f"{100 * stat['stdev'] / stat['mean']:.0f}"


def peak_rss_mb():
    """Kernel high-water mark of RSS for this process (Linux reports KiB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def rss_mb():
    return psutil.Process().memory_info().rss / MB


# --------------------------------------------------------------------------- benchmark


def build_prompt(engine, target_tokens):
    """Prompt whose tokenised length is ~target_tokens (before the chat template)."""
    if len(engine.tokenize(_PROMPT_HEAD)) >= target_tokens:
        return _PROMPT_HEAD

    def make(n):
        return _PROMPT_HEAD + " ".join(_FILLER_WORDS[i % len(_FILLER_WORDS)] for i in range(n))

    lo, hi = 0, target_tokens * 2
    while lo < hi:
        mid = (lo + hi) // 2
        if len(engine.tokenize(make(mid))) < target_tokens:
            lo = mid + 1
        else:
            hi = mid
    return make(lo)


def _chunk_has_output(chunk):
    for item in chunk.get("content") or []:
        if item.get("text"):
            return True
    return any(chunk.get("channels", {}).values())


def generate_once(litert_lm, engine, prompt, args, sampler):
    """One fresh conversation, one streamed generation. Returns a metrics dict."""
    sampler_cfg = litert_lm.SamplerConfig(top_k=1, seed=0) if args.greedy else None
    thinking_cfg = {
        "default": None,
        "on": litert_lm.ThinkingConfig(enable_thinking=True),
        "off": litert_lm.ThinkingConfig(enable_thinking=False),
    }[args.thinking]

    t_conv = time.perf_counter()
    with engine.create_conversation(
        sampler_config=sampler_cfg,
        thinking_config=thinking_cfg,
        max_output_tokens=args.max_output_tokens,
    ) as conv:
        conv_create_s = time.perf_counter() - t_conv

        chunk_times = []
        chars = 0
        t_start = time.perf_counter()
        for chunk in conv.send_message_async(prompt):
            if not _chunk_has_output(chunk):
                continue
            chunk_times.append(time.perf_counter())
            chars += sum(len(i.get("text", "")) for i in chunk.get("content") or [])
        t_end = time.perf_counter()

        info = conv.get_benchmark_info()
        context_tokens = conv.token_count

    ttft_wall = (chunk_times[0] - t_start) if chunk_times else None
    decode_n = info.last_decode_token_count
    wall_decode_tps = None
    if chunk_times and decode_n > 1 and t_end > chunk_times[0]:
        wall_decode_tps = (decode_n - 1) / (t_end - chunk_times[0])
    gaps_ms = [(b - a) * 1000 for a, b in zip(chunk_times, chunk_times[1:])]
    snap = sampler.latest()

    return {
        "conv_create_s": conv_create_s,
        "ttft_wall_s": ttft_wall,
        "ttft_engine_s": info.time_to_first_token_in_second,
        "prefill_tokens": info.last_prefill_token_count,
        "prefill_tps": info.last_prefill_tokens_per_second,
        "decode_tokens": decode_n,
        "decode_tps_engine": info.last_decode_tokens_per_second,
        "decode_tps_wall": wall_decode_tps,
        "chunks": len(chunk_times),
        "chunk_gap_p50_ms": percentile(gaps_ms, 50),
        "chunk_gap_p95_ms": percentile(gaps_ms, 95),
        "total_s": t_end - t_start,
        "output_chars": chars,
        "context_tokens": context_tokens,
        "temp_c": snap.get("temp_c"),
        "freq_mhz": snap.get("freq_mhz"),
        "rss_mb": rss_mb(),
        "peak_rss_mb": peak_rss_mb(),
    }


def drop_page_cache():
    os.sync()
    try:
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3\n")
        return True
    except OSError as e:
        print(f"  ! could not drop page cache ({e}); run as root for a true cold load")
        return False


def load_engine(litert_lm, args, sampler, results):
    """Load the model --load-runs times; the last engine is returned still open."""
    loads = []
    engine = None
    for i in range(args.load_runs):
        if engine is not None:
            engine.close()
            engine = None
            gc.collect()
            sampler.phase = f"unload{i}"
            time.sleep(1.0)
        rss_after_close = rss_mb()
        if i == 0 and args.drop_caches:
            drop_page_cache()

        sampler.phase = f"load{i + 1}"
        rss_before = rss_mb()
        t0 = time.perf_counter()
        engine = litert_lm.Engine(
            args.model,
            backend=litert_lm.Backend.CPU(thread_count=args.threads),
            max_num_tokens=args.max_num_tokens,
            cache_dir=args.cache_dir,
            enable_benchmark=True,
            enable_ynnpack=args.ynnpack,
        )
        load_s = time.perf_counter() - t0
        rec = {
            "load": i + 1,
            "kind": "first" if i == 0 else "reload",
            "load_s": load_s,
            "rss_before_mb": rss_before,
            "rss_after_mb": rss_mb(),
            "rss_delta_mb": rss_mb() - rss_before,
            "rss_after_previous_close_mb": rss_after_close if i else None,
            "peak_rss_mb": peak_rss_mb(),
            "temp_c": sampler.latest().get("temp_c"),
        }
        loads.append(rec)
        print(
            f"  load {rec['load']} ({rec['kind']}): {load_s:6.2f}s  "
            f"RSS +{rec['rss_delta_mb']:.0f} MB -> {rec['rss_after_mb']:.0f} MB"
        )
    results["load"] = loads
    return engine


def run_generation_phase(litert_lm, engine, args, sampler, results, runs_out):
    prompts = {}
    for target in args.prompt_tokens:
        prompts[target] = build_prompt(engine, target)
        print(f"  prompt target {target}: {len(engine.tokenize(prompts[target]))} tokens (pre-template)")

    if args.warmup:
        sampler.phase = "warmup"
        warm = prompts[min(prompts)]
        for i in range(args.warmup):
            m = generate_once(litert_lm, engine, warm, args, sampler)
            print(f"  warmup {i + 1}: {m['decode_tps_engine']:.2f} decode tok/s")

    per_size = {}
    for target, prompt in prompts.items():
        recs = []
        for i in range(args.runs):
            sampler.phase = f"run p{target} #{i + 1}"
            try:
                m = generate_once(litert_lm, engine, prompt, args, sampler)
            except Exception as e:  # keep going; a failed size shouldn't lose the rest
                print(f"  p{target} run {i + 1}: FAILED ({type(e).__name__}: {e})")
                runs_out.append({"phase": sampler.phase, "prompt_target": target, "error": str(e)})
                continue
            m.update(phase="generation", prompt_target=target, run=i + 1)
            recs.append(m)
            runs_out.append(m)
            print(
                f"  p{target} run {i + 1}: TTFT {m['ttft_wall_s']:.2f}s  "
                f"prefill {m['prefill_tps']:.1f} t/s  decode {m['decode_tps_engine']:.2f} t/s "
                f"({m['decode_tokens']} tok)  temp {fmt_temp(m['temp_c'])}"
            )
        per_size[target] = recs
    return per_size


def run_sustained(litert_lm, engine, args, sampler, prompt, runs_out):
    """Generate back-to-back for N minutes to expose thermal throttling."""
    deadline = time.perf_counter() + args.sustained_minutes * 60
    recs = []
    i = 0
    while time.perf_counter() < deadline:
        i += 1
        sampler.phase = "sustained"
        m = generate_once(litert_lm, engine, prompt, args, sampler)
        m.update(phase="sustained", prompt_target=args.prompt_tokens[len(args.prompt_tokens) // 2], run=i)
        recs.append(m)
        runs_out.append(m)
        print(
            f"  sustained {i}: decode {m['decode_tps_engine']:.2f} t/s  "
            f"temp {fmt_temp(m['temp_c'])}  freq {m['freq_mhz'] or 0:.0f} MHz"
        )
    return recs


def fmt_temp(t):
    return "n/a" if t is None else f"{t:.1f}C"


# --------------------------------------------------------------------------- reporting


def phase_group(name):
    """Collapse 'run p128 #2' -> 'generation', 'load2' -> 'load', etc."""
    if name.startswith("run ") or name == "warmup":
        return "generation"
    if name.startswith("load"):
        return "load"
    if name.startswith("unload"):
        return "unload"
    return name


def phase_table(samples):
    groups = {}
    for s in samples:
        groups.setdefault(phase_group(s["phase"]), []).append(s)
    table = {}
    for name, rows in groups.items():
        flags = 0
        for r in rows:
            flags |= r["throttled"] or 0
        table[name] = {
            "samples": len(rows),
            "temp_c": summarize([r["temp_c"] for r in rows]),
            "freq_mhz": summarize([r["freq_mhz"] for r in rows]),
            "sys_cpu_pct": summarize([r["sys_cpu_pct"] for r in rows]),
            "proc_cpu_pct": summarize([r["proc_cpu_pct"] for r in rows]),
            "rss_mb": summarize([r["rss_mb"] for r in rows]),
            "sys_avail_mb": summarize([r["sys_avail_mb"] for r in rows]),
            "swap_used_mb": summarize([r["swap_used_mb"] for r in rows]),
            "throttle_flags": decode_throttled(flags),
        }
    return table


def print_report(results, per_size, sustained, table):
    print("\n" + "=" * 96)
    print("RESULTS")
    print("=" * 96)

    print("\nModel load")
    print(f"  {'#':<3}{'kind':<8}{'load s':>9}{'RSS +MB':>10}{'RSS MB':>9}{'peak RSS':>10}")
    for r in results.get("load", []):
        print(
            f"  {r['load']:<3}{r['kind']:<8}{r['load_s']:>9.2f}{r['rss_delta_mb']:>10.0f}"
            f"{r['rss_after_mb']:>9.0f}{r['peak_rss_mb']:>10.0f}"
        )

    print("\nGeneration (mean over runs; decode tok/s = engine-reported)")
    print(
        f"  {'prompt':>7}{'prefill tok':>12}{'prefill t/s':>12}{'TTFT s':>9}"
        f"{'decode t/s':>12}{'wall t/s':>10}{'dec cv%':>9}{'gap p95 ms':>12}{'peak RSS':>10}"
    )
    for target, recs in per_size.items():
        if not recs:
            continue
        col = lambda k: summarize([r[k] for r in recs])  # noqa: E731
        print(
            f"  {target:>7}{fmt(col('prefill_tokens'), digits=0, width=12)}"
            f"{fmt(col('prefill_tps'), digits=1, width=12)}{fmt(col('ttft_wall_s'), width=9)}"
            f"{fmt(col('decode_tps_engine'), width=12)}{fmt(col('decode_tps_wall'), width=10)}"
            f"{cv_pct(col('decode_tps_engine')):>9}"
            f"{fmt(col('chunk_gap_p95_ms'), digits=1, width=12)}{fmt(col('peak_rss_mb'), digits=0, width=10)}"
        )

    if sustained:
        third = max(1, len(sustained) // 3)
        first = statistics.fmean(r["decode_tps_engine"] for r in sustained[:third])
        last = statistics.fmean(r["decode_tps_engine"] for r in sustained[-third:])
        print(
            f"\nSustained ({len(sustained)} runs): decode {first:.2f} t/s (first third) -> "
            f"{last:.2f} t/s (last third), {100 * (last - first) / first:+.1f}%"
        )

    print("\nSystem by phase")
    print(
        f"  {'phase':<11}{'temp avg':>9}{'temp max':>9}{'MHz avg':>9}{'MHz min':>9}"
        f"{'proc CPU%':>10}{'sys CPU%':>9}{'RSS max':>9}{'avail min':>10}"
    )
    for name, t in table.items():
        print(
            f"  {name:<11}{fmt(t['temp_c'], 'mean', 1, 9)}{fmt(t['temp_c'], 'max', 1, 9)}"
            f"{fmt(t['freq_mhz'], 'mean', 0, 9)}{fmt(t['freq_mhz'], 'min', 0, 9)}"
            f"{fmt(t['proc_cpu_pct'], 'mean', 0, 10)}{fmt(t['sys_cpu_pct'], 'mean', 0, 9)}"
            f"{fmt(t['rss_mb'], 'max', 0, 9)}{fmt(t['sys_avail_mb'], 'min', 0, 10)}"
        )

    warnings = []
    for target, recs in per_size.items():
        stat = summarize([r["decode_tps_engine"] for r in recs])
        if stat and stat["n"] > 1 and stat["stdev"] / stat["mean"] > UNSTABLE_CV:
            first, last = recs[0]["decode_tps_engine"], recs[-1]["decode_tps_engine"]
            warnings.append(
                f"p{target}: decode tok/s varies {cv_pct(stat)}% between runs "
                f"(first {first:.1f} -> last {last:.1f}); results are not steady-state, "
                "raise --warmup/--runs and check runs.csv"
            )
    for name, t in table.items():
        if t["throttle_flags"]:
            warnings.append(f"{name}: {', '.join(t['throttle_flags'])}")
        if t["swap_used_mb"] and t["swap_used_mb"]["max"] > 1:
            warnings.append(f"{name}: swap in use ({t['swap_used_mb']['max']:.0f} MB)")
    if warnings:
        print("\nWARNINGS")
        for w in warnings:
            print(f"  ! {w}")
    print()


def system_info(args, sensors):
    model = _read_text("/proc/device-tree/model")
    if model:
        model = model.rstrip("\x00")
    vm = psutil.virtual_memory()

    def version(pkg):
        try:
            return metadata.version(pkg)
        except metadata.PackageNotFoundError:
            return None

    return {
        "device": model or platform.node(),
        "machine": platform.machine(),
        "kernel": platform.release(),
        "python": platform.python_version(),
        "litert_lm_api": version("litert-lm-api"),
        "cpu_count": psutil.cpu_count(),
        "ram_total_mb": round(vm.total / MB),
        "ram_available_mb": round(vm.available / MB),
        "swap_total_mb": round(psutil.swap_memory().total / MB),
        "governor": _read_text("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "max_freq_mhz": (lambda v: int(v) / 1000 if v and v.isdigit() else None)(
            _read_text("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq")
        ),
        "sensors": sensors.describe(),
        "throttled_at_start": decode_throttled(sensors.throttled()),
        "model_file": os.path.abspath(args.model),
        "model_size_mb": round(os.path.getsize(args.model) / MB, 1),
    }


def write_outputs(out_dir, results, runs, samples):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    for name, rows in (("runs.csv", runs), ("samples.csv", samples)):
        if not rows:
            continue
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(os.path.join(out_dir, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    print(f"Wrote results to {out_dir}/ (summary.json, runs.csv, samples.csv)")


# --------------------------------------------------------------------------- main


def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark LiteRT-LM Gemma load/inference, CPU temperature and memory on a Raspberry Pi.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", required=True, help="path to a .litertlm model file")
    p.add_argument("--threads", type=int, default=None, help="CPU threads (default: library default)")
    p.add_argument("--ynnpack", action="store_true", help="enable the YNNPACK delegate (arm64)")
    p.add_argument("--max-num-tokens", type=int, default=None, help="KV-cache size (default: model default)")
    p.add_argument("--cache-dir", default=None, help="writable dir for compiled-artifact cache")
    p.add_argument("--load-runs", type=int, default=2, help="times to load the model (1st = first load, rest = reloads)")
    p.add_argument("--drop-caches", action="store_true", help="drop the page cache before the 1st load (needs root)")
    p.add_argument("--prompt-tokens", default="32,128,512", help="comma-separated prompt sizes to sweep")
    p.add_argument("--max-output-tokens", type=int, default=128, help="cap on generated tokens per run")
    p.add_argument("--runs", type=int, default=5, help="measured runs per prompt size")
    p.add_argument("--warmup", type=int, default=2, help="unmeasured warmup generations")
    p.add_argument("--sustained-minutes", type=float, default=0, help="extra back-to-back soak test (0 = skip)")
    p.add_argument("--idle-seconds", type=float, default=10, help="idle baseline before loading")
    p.add_argument("--cooldown-seconds", type=float, default=30, help="idle recording after the run")
    p.add_argument("--sample-interval", type=float, default=1.0, help="sensor sampling period (s)")
    p.add_argument("--greedy", action=argparse.BooleanOptionalAction, default=True, help="greedy decoding for repeatability")
    p.add_argument("--thinking", choices=("default", "on", "off"), default="default", help="thinking mode override")
    p.add_argument("--log-level", choices=("verbose", "info", "warning", "error", "silent"), default="error")
    p.add_argument("--out-dir", default=None, help="output dir (default: bench_results/<timestamp>)")
    args = p.parse_args()
    try:
        args.prompt_tokens = sorted({int(x) for x in args.prompt_tokens.split(",") if x.strip()})
    except ValueError:
        p.error("--prompt-tokens must be comma-separated integers")
    if not args.prompt_tokens:
        p.error("--prompt-tokens is empty")
    if not os.path.isfile(args.model):
        p.error(f"model file not found: {args.model}")
    if os.path.getsize(args.model) < MB:
        p.error(
            f"{args.model} is only {os.path.getsize(args.model)} bytes; the download is probably "
            "an error page (gated model? run `hf auth login`) or truncated"
        )
    if args.out_dir is None:
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        args.out_dir = os.path.join("bench_results", stamp)
    return args


def main():
    args = parse_args()
    try:
        import litert_lm
    except ImportError:
        sys.exit("litert_lm is required:  pip install litert-lm-api")
    litert_lm.set_min_log_severity(getattr(litert_lm.LogSeverity, args.log_level.upper()))

    sensors = Sensors()
    info = system_info(args, sensors)
    print("System")
    for k, v in info.items():
        print(f"  {k:<18} {v}")
    if info["swap_total_mb"] and info["swap_total_mb"] > 0:
        print("  ! swap is enabled; heavy swapping will distort results")
    if info["model_size_mb"] > info["ram_available_mb"]:
        print("  ! model file is larger than available RAM")

    results = {"system": info, "args": vars(args), "started": dt.datetime.now().isoformat(timespec="seconds")}
    runs = []
    sampler = Sampler(sensors, args.sample_interval)
    sampler.start()
    engine = None
    per_size, sustained = {}, []

    try:
        if args.idle_seconds > 0:
            print(f"\nIdle baseline ({args.idle_seconds:.0f}s)")
            sampler.phase = "idle"
            time.sleep(args.idle_seconds)

        print("\nLoading model")
        engine = load_engine(litert_lm, args, sampler, results)

        print("\nGeneration")
        per_size = run_generation_phase(litert_lm, engine, args, sampler, results, runs)

        if args.sustained_minutes > 0:
            print(f"\nSustained load ({args.sustained_minutes:g} min)")
            mid = args.prompt_tokens[len(args.prompt_tokens) // 2]
            sustained = run_sustained(litert_lm, engine, args, sampler, build_prompt(engine, mid), runs)

        if args.cooldown_seconds > 0:
            print(f"\nCooldown ({args.cooldown_seconds:.0f}s, model still loaded)")
            sampler.phase = "cooldown"
            time.sleep(args.cooldown_seconds)
    except KeyboardInterrupt:
        print("\nInterrupted; reporting what was collected so far.")
    finally:
        if engine is not None:
            engine.close()
        sampler.stop()

    table = phase_table(sampler.samples)
    results["generation"] = {
        str(t): {
            k: summarize([r[k] for r in recs])
            for k in (
                "ttft_wall_s", "ttft_engine_s", "prefill_tps", "decode_tps_engine",
                "decode_tps_wall", "decode_tokens", "chunk_gap_p95_ms", "total_s", "peak_rss_mb",
            )
        }
        for t, recs in per_size.items() if recs
    }
    results["phases"] = table
    results["peak_rss_mb"] = peak_rss_mb()
    print_report(results, per_size, sustained, table)
    write_outputs(args.out_dir, results, runs, sampler.samples)


if __name__ == "__main__":
    main()
