# bench_litert.py — LiteRT-LM Gemma benchmark for Raspberry Pi

Benchmarks a [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) model (e.g. Gemma-4 E2B)
running on the CPU of a Raspberry Pi, and records what the board is doing while it runs:

- **Model load** — load time and memory cost for a first load and for reloads
- **Latency and throughput** — time-to-first-token (TTFT), prefill tok/s and decode tok/s,
  both as reported by the engine and as measured by wall clock
- **Thermals** — CPU temperature, CPU frequency and the Pi firmware throttle flags
- **Resources** — process and system CPU %, RSS (resident memory), available RAM and swap

A background thread samples the sensors every `--sample-interval` seconds. Each sample is
tagged with the current phase (`idle`, `load1`, `run p128 #2`, `sustained`, `cooldown`, ...),
so you can line temperature and memory up against what the model was doing.

The script also runs on non-Pi Linux machines. Sensors that aren't available there (e.g.
`vcgencmd` throttle flags, or a VM's fake 0 °C thermal zone) are reported as `n/a`.

## Setup

```bash
python3 -m venv venv && . venv/bin/activate
pip install litert-lm-api psutil

# Download a model, e.g.:
hf download litert-community/gemma-4-E2B-it-litert-lm gemma-4-E2B-it.litertlm
```

If the model file is under 1 MB, the script refuses to run. That usually means the download
returned an error page (for a gated model, run `hf auth login` first).

## Usage

```bash
# Default run: 2 loads, prompts of 32/128/512 tokens, 5 runs each, 128 output tokens
python bench_litert.py --model gemma-4-E2B-it.litertlm

# Fix the thread count and sweep other prompt sizes
python bench_litert.py --model m.litertlm --threads 4 --prompt-tokens 32,256,1024

# True cold load (drops the page cache first; needs root)
sudo -E venv/bin/python bench_litert.py --model m.litertlm --drop-caches

# Thermal soak: generate back-to-back for 10 minutes
python bench_litert.py --model m.litertlm --sustained-minutes 10
```

Press Ctrl-C at any point to stop early. The script still reports and saves what it has
collected up to that point.

### Options

| Option | Default | Description |
|---|---|---|
| `--model` | *(required)* | Path to a `.litertlm` model file |
| `--threads` | library default | CPU threads for the engine |
| `--ynnpack` | off | Enable the YNNPACK delegate (arm64) |
| `--max-num-tokens` | model default | KV-cache size |
| `--cache-dir` | none | Writable dir for the compiled-artifact cache |
| `--load-runs` | 2 | Number of model loads (1st = first load, rest = reloads) |
| `--drop-caches` | off | Drop the page cache before the 1st load (needs root) |
| `--prompt-tokens` | `32,128,512` | Comma-separated prompt sizes to sweep |
| `--max-output-tokens` | 128 | Cap on generated tokens per run |
| `--runs` | 5 | Measured runs per prompt size |
| `--warmup` | 2 | Unmeasured warmup generations |
| `--sustained-minutes` | 0 | Extra back-to-back soak test (0 = skip) |
| `--idle-seconds` | 10 | Idle baseline recorded before loading |
| `--cooldown-seconds` | 30 | Idle recording after the run (model still loaded) |
| `--sample-interval` | 1.0 | Sensor sampling period in seconds |
| `--greedy` / `--no-greedy` | greedy | Greedy decoding (top-k 1, seed 0) for repeatable output |
| `--thinking` | `default` | Thinking mode override: `default`, `on`, `off` |
| `--log-level` | `error` | LiteRT-LM log level |
| `--out-dir` | `bench_results/<timestamp>` | Where results are written |

## What a run does

1. **Idle baseline:** records sensors with nothing loaded.
2. **Load:** loads the model `--load-runs` times and records load time and RSS change for each.
3. **Warmup:** runs a few unmeasured generations so caches and clocks settle.
4. **Generation sweep:** for each prompt size, builds a filler prompt of about that many
   tokens and runs `--runs` fresh conversations, streaming the output.
5. **Sustained (optional):** generates back-to-back with the middle prompt size to show
   thermal throttling. The report compares decode speed in the first and last third of the runs.
6. **Cooldown:** records sensors with the model still loaded.

At the end it prints a results table and warnings for: decode speed varying by more than 15%
between runs, throttle flags being set, or swap being used.

## Output files

All three files are written to `--out-dir`.

### `summary.json`
- `system`: device, kernel, Python and `litert-lm-api` versions, RAM, swap, CPU governor, max
  frequency, sensor sources, throttle state at start, model path and size
- `args`: the options the run used
- `load`: one record per model load
- `generation`: mean / stdev / min / p50 / max of each metric, per prompt size
- `phases`: sensor statistics grouped by phase (idle, load, generation, cooldown, ...)
- `peak_rss_mb`: peak RSS of the process over the whole run

### `runs.csv`: one row per measured generation

| Column | Meaning |
|---|---|
| `ttft_wall_s` / `ttft_engine_s` | Time to first token (wall clock / engine-reported) |
| `prefill_tokens`, `prefill_tps` | Prompt tokens after the chat template, and prefill speed |
| `decode_tokens`, `decode_tps_engine` | Generated tokens and engine-reported decode speed |
| `decode_tps_wall` | Decode speed measured from the first to the last streamed chunk |
| `chunk_gap_p50_ms`, `chunk_gap_p95_ms` | Time between streamed chunks (how smooth the stream is) |
| `total_s` | Wall time for the whole request (prefill + decode) |
| `context_tokens` | Tokens in the conversation at the end |
| `temp_c`, `freq_mhz` | Latest sensor reading when the run finished |
| `rss_mb`, `peak_rss_mb` | Current and peak resident memory |
| `prompt_target`, `run`, `phase` | Which prompt size and run this row is |

### `samples.csv`: one row per sensor sample
`t_s`, `phase`, `temp_c`, `freq_mhz`, `sys_cpu_pct`, `proc_cpu_pct` (100 = one full core),
`rss_mb`, `sys_used_mb`, `sys_avail_mb`, `swap_used_mb`, `throttled` (the raw `vcgencmd get_throttled` value).

### RSS
RSS (Resident Set Size) is the part of the process's memory that is actually in physical RAM.
It includes shared libraries and memory-mapped model pages. The kernel can reclaim some of
those, so RSS can make memory use look higher than the true private cost.

## Example results: Raspberry Pi 5 (8 GB), Gemma-4 E2B

These results are from a run on 2026-09-21 with default options, stock cooling and the
`ondemand` governor. The files are `summary.json`, `runs.csv` and `samples.csv` in this directory.

| Prompt tokens | TTFT (s) | Prefill tok/s | Decode tok/s | Total per request (s) | Peak RSS (MB) |
|---|---|---|---|---|---|
| 32 | 1.26 | 40.0 | 8.72 | 15.7 | 2105 |
| 128 | 2.52 | 60.6 | 8.29 | 17.7 | 2149 |
| 512 | 6.40 | 84.8 | 7.99 | 22.2 | 2165 |

- Loading takes under 0.4 s; LiteRT-LM appears to memory-map the weights.
- RSS goes from about 470 MB after loading to about 2.1 GB once generation starts, and no
  swap was used.
- During generation the CPU ran at 2400 MHz, at about 362% process CPU (out of 400%), and
  peaked at 85.9 °C. No throttle flags were set.
