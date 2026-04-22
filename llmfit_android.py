#!/usr/bin/env python3
"""
llmfit-android — Find which LLMs run on your Android device (Termux)
Author: Built for Termux | Python 3.8+
"""

import os
import sys
import json
import subprocess
import math

# ── Optional rich UI ────────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODEL DATABASE
# Format: name, params_b, quant → ram_gb, quality, use_case, provider
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODELS = [
    # (name, provider, params_B, quant, ram_gb_needed, quality_score, use_case, runner)
    ("SmolLM2-135M",        "HuggingFace",  0.135, "Q4_K_M", 0.2,  40, "chat",      "llama.cpp"),
    ("SmolLM2-360M",        "HuggingFace",  0.36,  "Q4_K_M", 0.4,  50, "chat",      "llama.cpp"),
    ("Qwen2.5-0.5B",        "Alibaba",      0.5,   "Q4_K_M", 0.5,  52, "chat",      "llama.cpp"),
    ("Phi-3-mini-3.8B",     "Microsoft",    3.8,   "Q2_K",   1.8,  62, "reasoning", "llama.cpp"),
    ("Phi-3-mini-3.8B",     "Microsoft",    3.8,   "Q4_K_M", 2.8,  70, "reasoning", "llama.cpp"),
    ("Gemma-2B",            "Google",       2.0,   "Q4_K_M", 1.5,  60, "general",   "llama.cpp"),
    ("Gemma-2B",            "Google",       2.0,   "Q8_0",   2.2,  65, "general",   "llama.cpp"),
    ("Gemma-3-1B",          "Google",       1.0,   "Q4_K_M", 0.9,  58, "general",   "llama.cpp"),
    ("Gemma-3-4B",          "Google",       4.0,   "Q4_K_M", 3.0,  72, "general",   "llama.cpp"),
    ("Llama-3.2-1B",        "Meta",         1.0,   "Q4_K_M", 0.9,  60, "general",   "llama.cpp"),
    ("Llama-3.2-3B",        "Meta",         3.0,   "Q4_K_M", 2.2,  68, "general",   "llama.cpp"),
    ("Llama-3.2-3B",        "Meta",         3.0,   "Q8_0",   3.4,  73, "general",   "llama.cpp"),
    ("Llama-3.1-8B",        "Meta",         8.0,   "Q2_K",   3.5,  70, "general",   "llama.cpp"),
    ("Llama-3.1-8B",        "Meta",         8.0,   "Q4_K_M", 5.5,  78, "general",   "llama.cpp"),
    ("Llama-3.1-8B",        "Meta",         8.0,   "Q8_0",   9.0,  82, "general",   "llama.cpp"),
    ("Mistral-7B-v0.3",     "Mistral AI",   7.0,   "Q2_K",   3.2,  70, "general",   "llama.cpp"),
    ("Mistral-7B-v0.3",     "Mistral AI",   7.0,   "Q4_K_M", 4.8,  77, "general",   "llama.cpp"),
    ("Mistral-7B-v0.3",     "Mistral AI",   7.0,   "Q8_0",   7.7,  81, "general",   "llama.cpp"),
    ("TinyLlama-1.1B",      "StatNLP",      1.1,   "Q4_K_M", 0.9,  55, "chat",      "llama.cpp"),
    ("TinyLlama-1.1B",      "StatNLP",      1.1,   "Q8_0",   1.2,  58, "chat",      "llama.cpp"),
    ("Qwen2.5-1.5B",        "Alibaba",      1.5,   "Q4_K_M", 1.2,  62, "coding",    "llama.cpp"),
    ("Qwen2.5-3B",          "Alibaba",      3.0,   "Q4_K_M", 2.3,  70, "coding",    "llama.cpp"),
    ("Qwen2.5-7B",          "Alibaba",      7.0,   "Q4_K_M", 4.9,  78, "coding",    "llama.cpp"),
    ("Qwen2.5-Coder-1.5B",  "Alibaba",      1.5,   "Q4_K_M", 1.2,  65, "coding",    "llama.cpp"),
    ("Qwen2.5-Coder-3B",    "Alibaba",      3.0,   "Q4_K_M", 2.3,  72, "coding",    "llama.cpp"),
    ("Qwen2.5-Coder-7B",    "Alibaba",      7.0,   "Q4_K_M", 5.0,  80, "coding",    "llama.cpp"),
    ("DeepSeek-R1-1.5B",    "DeepSeek",     1.5,   "Q4_K_M", 1.2,  68, "reasoning", "llama.cpp"),
    ("DeepSeek-R1-7B",      "DeepSeek",     7.0,   "Q4_K_M", 5.0,  80, "reasoning", "llama.cpp"),
    ("Phi-2",               "Microsoft",    2.7,   "Q4_K_M", 2.0,  65, "coding",    "llama.cpp"),
    ("Phi-2",               "Microsoft",    2.7,   "Q8_0",   3.0,  68, "coding",    "llama.cpp"),
    ("OpenHermes-2.5-7B",   "Teknium",      7.0,   "Q4_K_M", 4.8,  76, "chat",      "llama.cpp"),
    ("Neural-Chat-7B",      "Intel",        7.0,   "Q4_K_M", 4.8,  74, "chat",      "llama.cpp"),
    ("Orca-Mini-3B",        "Microsoft",    3.0,   "Q4_K_M", 2.2,  66, "reasoning", "llama.cpp"),
    ("StableLM-2-1.6B",     "Stability AI", 1.6,   "Q4_K_M", 1.2,  60, "general",   "llama.cpp"),
    ("Falcon-RW-1B",        "TII UAE",      1.0,   "Q4_K_M", 0.9,  55, "general",   "llama.cpp"),
]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHIPSET DATABASE  (soc_keyword → approx shared_vram_gb, gpu_score)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHIPSET_DB = {
    # Snapdragon
    "sm8750": ("Snapdragon 8 Elite",    4.0, 95),
    "sm8650": ("Snapdragon 8 Gen 3",    3.0, 90),
    "sm8550": ("Snapdragon 8 Gen 2",    2.5, 82),
    "sm8475": ("Snapdragon 8+ Gen 1",   2.0, 75),
    "sm8450": ("Snapdragon 8 Gen 1",    2.0, 72),
    "sm8350": ("Snapdragon 888",        1.5, 65),
    "sm8250": ("Snapdragon 865",        1.5, 60),
    "sm7675": ("Snapdragon 7s Gen 3",   1.5, 68),
    "sm7550": ("Snapdragon 7 Gen 3",    1.5, 65),
    "sm7450": ("Snapdragon 7 Gen 1",    1.0, 58),
    "sm6375": ("Snapdragon 695",        0.8, 45),
    "sm6350": ("Snapdragon 690",        0.8, 42),
    # Dimensity
    "mt6989": ("Dimensity 9300+",       3.0, 88),
    "mt6985": ("Dimensity 9300",        3.0, 85),
    "mt6983": ("Dimensity 9200+",       2.5, 80),
    "mt6982": ("Dimensity 9200",        2.5, 78),
    "mt6979": ("Dimensity 9000+",       2.0, 72),
    "mt6977": ("Dimensity 9000",        2.0, 70),
    "mt6896": ("Dimensity 1200",        1.5, 60),
    "mt6893": ("Dimensity 1100",        1.5, 58),
    "mt6891": ("Dimensity 1000+",       1.5, 56),
    # Exynos
    "exynos2400": ("Exynos 2400",       2.5, 82),
    "exynos2200": ("Exynos 2200",       2.0, 72),
    "exynos2100": ("Exynos 2100",       1.5, 65),
    "exynos990":  ("Exynos 990",        1.2, 55),
    # Kirin
    "kirin9010": ("Kirin 9010",         2.0, 70),
    "kirin9000": ("Kirin 9000",         2.0, 68),
    "kirin990":  ("Kirin 990",          1.5, 58),
    # Apple (rare but possible via emulation)
    "a17":  ("Apple A17 Pro",           4.0, 98),
    "a16":  ("Apple A16 Bionic",        3.5, 93),
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HARDWARE DETECTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def getprop(key):
    try:
        r = subprocess.run(["getprop", key], capture_output=True, text=True, timeout=3)
        return r.stdout.strip()
    except Exception:
        return ""

def read_file(path):
    try:
        with open(path) as f:
            return f.read()
    except Exception:
        return ""

def detect_ram():
    mem = read_file("/proc/meminfo")
    for line in mem.splitlines():
        if line.startswith("MemTotal:"):
            kb = int(line.split()[1])
            return round(kb / 1024 / 1024, 1)  # GB
    return 2.0  # fallback

def detect_available_ram():
    mem = read_file("/proc/meminfo")
    avail = 0
    for line in mem.splitlines():
        if line.startswith("MemAvailable:"):
            kb = int(line.split()[1])
            avail = round(kb / 1024 / 1024, 1)
    return avail

def detect_cpu():
    cpu_info = read_file("/proc/cpuinfo")
    cores = cpu_info.count("processor\t:")
    arch = "unknown"
    hardware = ""
    for line in cpu_info.splitlines():
        if "Hardware" in line:
            hardware = line.split(":")[-1].strip()
        if "model name" in line.lower() or "Processor" in line:
            arch = line.split(":")[-1].strip()
    if not arch or arch == "unknown":
        arch = hardware or "ARM"
    # Count cores from nproc if available
    try:
        r = subprocess.run(["nproc"], capture_output=True, text=True, timeout=2)
        cores = int(r.stdout.strip())
    except Exception:
        pass
    return cores or 4, arch[:40]

def detect_chipset():
    """Try multiple methods to detect SoC."""
    candidates = [
        getprop("ro.board.platform"),
        getprop("ro.hardware"),
        getprop("ro.chipname"),
        getprop("ro.product.board"),
    ]
    soc_raw = ""
    for c in candidates:
        if c:
            soc_raw = c.lower()
            break

    # Match against DB
    for key, val in CHIPSET_DB.items():
        if key in soc_raw:
            return val[0], val[1], val[2], soc_raw

    # Fuzzy match brand names
    brand_map = {
        "snapdragon": ("Snapdragon (unknown)", 1.5, 60),
        "kirin":      ("Kirin (unknown)",       1.2, 55),
        "dimensity":  ("Dimensity (unknown)",   1.5, 58),
        "exynos":     ("Exynos (unknown)",      1.2, 55),
        "helio":      ("Helio (unknown)",       0.8, 40),
        "tensor":     ("Google Tensor",         1.5, 65),
    }
    model_name = getprop("ro.product.model").lower()
    combined = soc_raw + " " + model_name
    for brand, info in brand_map.items():
        if brand in combined:
            return info[0], info[1], info[2], soc_raw

    return "Unknown SoC", 1.0, 50, soc_raw

def detect_android_version():
    return getprop("ro.build.version.release") or "Unknown"

def detect_device_name():
    brand = getprop("ro.product.brand") or ""
    model = getprop("ro.product.model") or ""
    return f"{brand} {model}".strip() or "Android Device"

def detect_cpu_arch():
    abi = getprop("ro.product.cpu.abi") or ""
    if "arm64" in abi or "aarch64" in abi:
        return "ARM64 (64-bit)"
    elif "armeabi" in abi:
        return "ARM32 (32-bit)"
    elif "x86_64" in abi:
        return "x86_64"
    return abi or "ARM"

def gather_hardware():
    total_ram   = detect_ram()
    avail_ram   = detect_available_ram()
    cores, cpu_model = detect_cpu()
    soc_name, gpu_vram_est, gpu_score, soc_raw = detect_chipset()
    android_ver = detect_android_version()
    device_name = detect_device_name()
    cpu_arch    = detect_cpu_arch()

    # Usable RAM for LLMs: conservatively 60% of available, or 50% of total
    usable_ram = min(avail_ram * 0.85, total_ram * 0.65)
    usable_ram = round(usable_ram, 1)

    return {
        "device":       device_name,
        "android":      android_ver,
        "soc":          soc_name,
        "soc_raw":      soc_raw,
        "cpu_arch":     cpu_arch,
        "cpu_cores":    cores,
        "cpu_model":    cpu_model,
        "total_ram_gb": total_ram,
        "avail_ram_gb": avail_ram,
        "usable_ram_gb":usable_ram,
        "gpu_vram_est": gpu_vram_est,
        "gpu_score":    gpu_score,
    }

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SCORING ENGINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def fit_label(ram_needed, usable_ram):
    ratio = usable_ram / ram_needed if ram_needed > 0 else 0
    if ratio >= 1.3:
        return "PERFECT", 4
    elif ratio >= 1.0:
        return "GOOD", 3
    elif ratio >= 0.85:
        return "TIGHT", 2
    else:
        return "NO FIT", 1

def estimate_tps(ram_needed, usable_ram, cores, gpu_score, params_b):
    """Rough tokens/sec estimate for ARM CPU inference."""
    if usable_ram < ram_needed * 0.85:
        return 0
    # Base speed from cores and quantization pressure
    base = (cores / 8) * 8.0
    # Penalty for large models
    size_penalty = max(0.2, 1.0 - (params_b - 1) * 0.07)
    # RAM headroom bonus
    headroom = min(1.5, usable_ram / max(ram_needed, 0.1))
    # GPU assist bonus (shared memory, not VRAM)
    gpu_bonus = 1.0 + (gpu_score / 200)
    tps = base * size_penalty * headroom * gpu_bonus
    return max(0, round(tps, 1))

def score_models(hw):
    results = []
    seen = set()
    for row in MODELS:
        name, provider, params_b, quant, ram_gb, quality, use_case, runner = row
        key = (name, quant)
        if key in seen:
            continue
        seen.add(key)

        fit, fit_rank = fit_label(ram_gb, hw["usable_ram_gb"])
        tps = estimate_tps(ram_gb, hw["usable_ram_gb"], hw["cpu_cores"], hw["gpu_score"], params_b)

        # Composite score: quality + fit + speed
        composite = (quality * 0.5) + (fit_rank * 15) + min(tps * 0.5, 20)

        results.append({
            "name":      name,
            "provider":  provider,
            "params":    params_b,
            "quant":     quant,
            "ram_gb":    ram_gb,
            "quality":   quality,
            "use_case":  use_case,
            "runner":    runner,
            "fit":       fit,
            "fit_rank":  fit_rank,
            "tps":       tps,
            "score":     round(composite, 1),
        })

    results.sort(key=lambda x: (-x["fit_rank"], -x["score"]))
    return results

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DISPLAY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FIT_COLORS = {
    "PERFECT": "bright_green",
    "GOOD":    "green",
    "TIGHT":   "yellow",
    "NO FIT":  "red",
}

FIT_EMOJI = {
    "PERFECT": "✅",
    "GOOD":    "🟢",
    "TIGHT":   "🟡",
    "NO FIT":  "❌",
}

def print_header_rich(hw):
    lines = [
        f"[bold cyan]Device   :[/bold cyan]  {hw['device']}",
        f"[bold cyan]Android  :[/bold cyan]  {hw['android']}",
        f"[bold cyan]SoC      :[/bold cyan]  {hw['soc']}",
        f"[bold cyan]CPU      :[/bold cyan]  {hw['cpu_cores']} cores · {hw['cpu_arch']}",
        f"[bold cyan]RAM      :[/bold cyan]  {hw['total_ram_gb']} GB total  |  {hw['avail_ram_gb']} GB free  |  [bold]{hw['usable_ram_gb']} GB usable for LLMs[/bold]",
        f"[bold cyan]GPU est  :[/bold cyan]  ~{hw['gpu_vram_est']} GB shared VRAM  (score {hw['gpu_score']}/100)",
    ]
    panel = Panel("\n".join(lines), title="[bold white]⚡ llmfit-android — System Info[/bold white]", border_style="cyan")
    console.print(panel)
    console.print()

def print_table_rich(models, limit=None, fit_filter=None, use_case=None):
    data = models
    if fit_filter:
        data = [m for m in data if m["fit"] == fit_filter.upper()]
    if use_case:
        data = [m for m in data if m["use_case"] == use_case.lower()]
    if limit:
        data = data[:limit]

    table = Table(
        title="[bold]Model Recommendations[/bold]",
        box=box.ROUNDED,
        show_lines=True,
        header_style="bold magenta",
    )
    table.add_column("#",         style="dim",          width=3)
    table.add_column("Model",     style="bold white",   min_width=20)
    table.add_column("Provider",  style="cyan",         width=12)
    table.add_column("Params",    justify="right",      width=7)
    table.add_column("Quant",     style="yellow",       width=8)
    table.add_column("RAM",       justify="right",      width=6)
    table.add_column("Fit",       justify="center",     width=9)
    table.add_column("~tok/s",    justify="right",      width=7)
    table.add_column("Use",       style="dim",          width=10)
    table.add_column("Runner",    style="dim",          width=10)

    for i, m in enumerate(data, 1):
        fit_color = FIT_COLORS.get(m["fit"], "white")
        fit_str   = f"[{fit_color}]{FIT_EMOJI[m['fit']]} {m['fit']}[/{fit_color}]"
        tps_str   = f"[green]{m['tps']}[/green]" if m["tps"] > 0 else "[red]—[/red]"
        table.add_row(
            str(i),
            m["name"],
            m["provider"],
            f"{m['params']}B",
            m["quant"],
            f"{m['ram_gb']}G",
            fit_str,
            tps_str,
            m["use_case"],
            m["runner"],
        )

    console.print(table)
    console.print(f"\n[dim]Showing {len(data)} models  |  tok/s = estimated CPU tokens/sec[/dim]\n")

# ── Fallback plain-text output ───────────────────────────────────────────────

def print_header_plain(hw):
    sep = "─" * 60
    print(sep)
    print("  llmfit-android — System Info")
    print(sep)
    print(f"  Device  : {hw['device']}")
    print(f"  Android : {hw['android']}")
    print(f"  SoC     : {hw['soc']}")
    print(f"  CPU     : {hw['cpu_cores']} cores · {hw['cpu_arch']}")
    print(f"  RAM     : {hw['total_ram_gb']} GB total | {hw['avail_ram_gb']} GB free | {hw['usable_ram_gb']} GB usable")
    print(f"  GPU est : ~{hw['gpu_vram_est']} GB shared VRAM")
    print(sep)

def print_table_plain(models, limit=None, fit_filter=None, use_case=None):
    data = models
    if fit_filter:
        data = [m for m in data if m["fit"] == fit_filter.upper()]
    if use_case:
        data = [m for m in data if m["use_case"] == use_case.lower()]
    if limit:
        data = data[:limit]

    fmt = "{:<3} {:<22} {:<12} {:<6} {:<8} {:<6} {:<9} {:<7} {:<10}"
    print(fmt.format("#", "Model", "Provider", "Params", "Quant", "RAM", "Fit", "~tok/s", "Use-case"))
    print("─" * 90)
    for i, m in enumerate(data, 1):
        emoji = FIT_EMOJI.get(m["fit"], "?")
        tps = str(m["tps"]) if m["tps"] > 0 else "—"
        print(fmt.format(
            i, m["name"][:22], m["provider"][:12],
            f"{m['params']}B", m["quant"],
            f"{m['ram_gb']}G", f"{emoji}{m['fit'][:7]}",
            tps, m["use_case"]
        ))
    print(f"\nShowing {len(data)} models | tok/s = estimated CPU tokens/sec")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HELP = """
Usage: python llmfit_android.py [command] [options]

Commands:
  (none)             Show all models ranked by fit (default)
  system             Show detected hardware info only
  recommend          Top recommendations (default 10)
  fit                Show only fitting models (PERFECT + GOOD)
  search <name>      Search model by name

Options:
  -n <num>           Limit results (e.g. -n 5)
  --perfect          Only PERFECT fit models
  --use-case <type>  Filter: general | coding | chat | reasoning
  --json             Output as JSON
  --help             Show this help

Examples:
  python llmfit_android.py
  python llmfit_android.py recommend -n 5
  python llmfit_android.py fit --perfect
  python llmfit_android.py --use-case coding -n 5
  python llmfit_android.py search llama
  python llmfit_android.py system
  python llmfit_android.py recommend --json
"""

def parse_args():
    args = sys.argv[1:]
    cmd       = "all"
    limit     = None
    perfect   = False
    use_case  = None
    as_json   = False
    search_q  = None

    i = 0
    while i < len(args):
        a = args[i]
        if a in ("--help", "-h"):
            print(HELP); sys.exit(0)
        elif a == "system":
            cmd = "system"
        elif a == "recommend":
            cmd = "recommend"
        elif a == "fit":
            cmd = "fit"
        elif a == "search" and i+1 < len(args):
            cmd = "search"; i += 1; search_q = args[i]
        elif a == "-n" and i+1 < len(args):
            i += 1; limit = int(args[i])
        elif a == "--perfect":
            perfect = True
        elif a == "--use-case" and i+1 < len(args):
            i += 1; use_case = args[i]
        elif a == "--json":
            as_json = True
        i += 1

    return cmd, limit, perfect, use_case, as_json, search_q

def main():
    cmd, limit, perfect, use_case, as_json, search_q = parse_args()

    hw = gather_hardware()
    models = score_models(hw)

    if cmd == "system":
        if as_json:
            print(json.dumps(hw, indent=2))
        elif HAS_RICH:
            print_header_rich(hw)
        else:
            print_header_plain(hw)
        return

    if cmd == "search":
        q = search_q.lower()
        models = [m for m in models if q in m["name"].lower() or q in m["provider"].lower()]

    if cmd == "recommend":
        limit = limit or 10

    if cmd == "fit" or perfect:
        models = [m for m in models if m["fit"] in ("PERFECT", "GOOD")]

    if use_case:
        models = [m for m in models if m["use_case"] == use_case.lower()]

    if limit:
        models = models[:limit]

    if as_json:
        out = {"hardware": hw, "models": models}
        print(json.dumps(out, indent=2))
        return

    if HAS_RICH:
        print_header_rich(hw)
        # rebuild with filters already applied
        table = Table(
            title="[bold]Model Recommendations[/bold]",
            box=box.ROUNDED,
            show_lines=True,
            header_style="bold magenta",
        )
        table.add_column("#",        style="dim",        width=3)
        table.add_column("Model",    style="bold white", min_width=20)
        table.add_column("Provider", style="cyan",       width=14)
        table.add_column("Params",   justify="right",    width=7)
        table.add_column("Quant",    style="yellow",     width=8)
        table.add_column("RAM req",  justify="right",    width=8)
        table.add_column("Fit",      justify="center",   width=10)
        table.add_column("~tok/s",   justify="right",    width=7)
        table.add_column("Use-case", style="dim",        width=10)
        table.add_column("Runner",   style="dim",        width=10)

        for i, m in enumerate(models, 1):
            fc  = FIT_COLORS.get(m["fit"], "white")
            fs  = f"[{fc}]{FIT_EMOJI[m['fit']]} {m['fit']}[/{fc}]"
            tps = f"[green]{m['tps']}[/green]" if m["tps"] > 0 else "[red]—[/red]"
            table.add_row(
                str(i), m["name"], m["provider"],
                f"{m['params']}B", m["quant"],
                f"{m['ram_gb']} GB", fs, tps,
                m["use_case"], m["runner"],
            )
        console.print(table)
        console.print(f"\n[dim]Showing {len(models)} models  |  tok/s = estimated CPU inference speed[/dim]")
        console.print("[dim]Tip: Install [bold]rich[/bold] for colors → pip install rich[/dim]\n")
    else:
        print_header_plain(hw)
        print_table_plain(models)

    # Always show runner tips
    if not as_json:
        print("\n── How to run models ──────────────────────────────────────────")
        print("  llama.cpp : pkg install clang cmake && git clone https://github.com/ggerganov/llama.cpp")
        print("  Termux    : pkg install python")
        print("  Models    : Download GGUF files from huggingface.co/models")
        print("  Tip       : Use 4-bit quants (Q4_K_M) for best RAM/quality balance")
        print("──────────────────────────────────────────────────────────────\n")

if __name__ == "__main__":
    main()
