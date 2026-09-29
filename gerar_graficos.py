#!/usr/bin/env python3
"""
Gera graficos de analise de desempenho do KNN OpenMP.

Uso:
    python3 gerar_graficos.py [csv_benchmark] [vtune_reports_dir] [output_dir]

Defaults: auto-detecta os arquivos disponiveis.
"""

import sys
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ---------------------------------------------------------------------------
# Parametros
# ---------------------------------------------------------------------------
CSV_BENCHMARK = sys.argv[1] if len(sys.argv) > 1 else None
VTUNE_DIR = sys.argv[2] if len(sys.argv) > 2 else None
OUTPUT_DIR = sys.argv[3] if len(sys.argv) > 3 else "graficos"

if CSV_BENCHMARK is None:
    for f in ["resultados_knn_sem_vtune.csv", "resultados_knn.csv"]:
        if os.path.exists(f):
            CSV_BENCHMARK = f
            break

if VTUNE_DIR is None:
    for d in ["vtune-completo/reports", "vtune-selected/reports", "test-vtune/reports"]:
        if os.path.isdir(d):
            VTUNE_DIR = d
            break

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"CSV benchmark: {CSV_BENCHMARK}")
print(f"VTune reports: {VTUNE_DIR}")
print(f"Output:        {OUTPUT_DIR}/")
print()

# ---------------------------------------------------------------------------
# Configuracao global de fontes
# ---------------------------------------------------------------------------
LEGEND_FONTSIZE = 16
LABEL_FONTSIZE = 18
TICK_FONTSIZE = 15
FIGSIZE = (12, 7)
FIGSIZE_BAR = (12, 7)
DPI = 150

# Estilos visuais: cada combinacao tem cor, marker e linha UNICOS
# Vermelho reservado SOMENTE para o sequencial
STYLES = {
    'static ON':  {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-',  'label': 'static simd=ON'},
    'static OFF': {'color': '#9467bd', 'marker': 's', 'linestyle': '--', 'label': 'static simd=OFF'},
    'dynamic ON':  {'color': '#2ca02c', 'marker': '^', 'linestyle': '-',  'label': 'dynamic simd=ON'},
    'dynamic OFF': {'color': '#e377c2', 'marker': 'v', 'linestyle': '--', 'label': 'dynamic simd=OFF'},
    'guided ON':  {'color': '#ff7f0e', 'marker': 'D', 'linestyle': '-',  'label': 'guided simd=ON'},
    'guided OFF': {'color': '#8c564b', 'marker': 'X', 'linestyle': '--', 'label': 'guided simd=OFF'},
}

VTUNE_STYLES = {
    'dynamic_simd':   {'color': '#2ca02c', 'marker': '^', 'linestyle': '-',  'label': 'dynamic simd=ON'},
    'dynamic_nosimd': {'color': '#e377c2', 'marker': 'v', 'linestyle': '--', 'label': 'dynamic simd=OFF'},
    'static_simd':    {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-',  'label': 'static simd=ON'},
    'static_nosimd':  {'color': '#9467bd', 'marker': 's', 'linestyle': '--', 'label': 'static simd=OFF'},
    'guided_simd':    {'color': '#ff7f0e', 'marker': 'D', 'linestyle': '-',  'label': 'guided simd=ON'},
    'guided_nosimd':  {'color': '#8c564b', 'marker': 'X', 'linestyle': '--', 'label': 'guided simd=OFF'},
}

SEQ_COLOR = '#d62728'


def get_style(variante):
    key = variante.replace('simd=', '')
    return STYLES.get(key, {'color': '#333', 'marker': 'o', 'linestyle': '-', 'label': variante})


def get_vtune_variante(tag):
    parts = tag.replace('par_', '').split('_')
    if 'nosimd' in tag:
        return parts[0] + '_nosimd'
    return parts[0] + '_simd'


def plot_vtune_lines(ax, par_data, y_col):
    for var in VTUNE_STYLES:
        mask = par_data['tag'].apply(lambda t: get_vtune_variante(t) == var)
        data = par_data[mask].sort_values('threads')
        if len(data) > 0:
            s = VTUNE_STYLES[var]
            ax.plot(data['threads'], data[y_col], marker=s['marker'], color=s['color'],
                    linestyle=s['linestyle'], linewidth=2, markersize=9, label=s['label'])


def save_fig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  -> {path}")


def parse_vtune_summary(filepath):
    metrics = {}
    with open(filepath, 'r') as fh:
        for line in fh:
            if '\t' in line and 'vtune:' not in line:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    name = parts[1].strip()
                    val_str = parts[2].strip()
                    try:
                        metrics[name] = float(val_str)
                    except ValueError:
                        m = re.match(r'([\d.]+)%', val_str)
                        if m:
                            metrics[name] = float(m.group(1))
                        else:
                            metrics[name] = val_str
    return metrics


def parse_vtune_hotspots(filepath):
    functions = []
    with open(filepath, 'r') as fh:
        header_found = False
        for line in fh:
            if 'Function' in line and 'CPU Time' in line and not header_found:
                header_found = True
                continue
            if header_found and '\t' in line and 'vtune:' not in line:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    try:
                        cpu_time = float(parts[1])
                        if cpu_time > 0:
                            functions.append({'Function': parts[0], 'CPU Time': cpu_time})
                    except ValueError:
                        pass
    return functions


def extract_tag_info(tag):
    if tag.startswith('seq'):
        return {'schedule': '-', 'simd': '-', 'threads': 1, 'label': 'SEQ'}
    parts = tag.split('_')
    schedule = parts[0] if parts else 'unknown'
    simd = 'ON' if 'simd' in tag and 'nosimd' not in tag else 'OFF'
    threads = 1
    for p in parts:
        m = re.match(r'(\d+)T', p)
        if m:
            threads = int(m.group(1))
    return {'schedule': schedule, 'simd': simd, 'threads': threads, 'label': f"{schedule}_{simd}_{threads}T"}


# ===========================================================================
# 1. Graficos de Benchmark CSV
# ===========================================================================
if CSV_BENCHMARK and os.path.exists(CSV_BENCHMARK):
    df = pd.read_csv(CSV_BENCHMARK)
    print(f"Benchmark: {len(df)} linhas carregadas")

    seq = df[df['versao'] == 'sequencial'][['N', 'tempo_seg']].rename(columns={'tempo_seg': 'tempo_seq'})
    par = df[df['versao'] == 'paralelo'].copy()
    par = par.merge(seq, on='N')
    par['speedup'] = par['tempo_seq'] / par['tempo_seg']
    par['eficiencia'] = par['speedup'] / par['threads']
    par['variante'] = par['schedule'] + ' simd=' + par['simd']
    n_max = par['N'].max()
    max_t = par['threads'].max()

    # --- Speedup por N (dynamic, simd ON) ---
    fig, ax = plt.subplots(figsize=FIGSIZE)
    subset = par[(par['schedule'] == 'dynamic') & (par['simd'] == 'ON')]
    for n_val in sorted(subset['N'].unique()):
        data = subset[subset['N'] == n_val].sort_values('threads')
        ax.plot(data['threads'], data['speedup'], 'o-', label=f'N={n_val}')
    ax.plot([1, max_t], [1, max_t], 'k--', alpha=0.3, label='Ideal')
    ax.set_xlabel('Threads', fontsize=LABEL_FONTSIZE)
    ax.set_ylabel('Speedup', fontsize=LABEL_FONTSIZE)
    ax.legend(fontsize=LEGEND_FONTSIZE)
    ax.tick_params(labelsize=TICK_FONTSIZE)
    ax.grid(True, alpha=0.3)
    save_fig(fig, f'{OUTPUT_DIR}/benchmark_speedup_por_N_dynamic_simdON.png')

    # --- Eficiencia por N (dynamic, simd ON) ---
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for n_val in sorted(subset['N'].unique()):
        data = subset[subset['N'] == n_val].sort_values('threads')
        ax.plot(data['threads'], data['eficiencia'], 'o-', label=f'N={n_val}')
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.3, label='Ideal')
    ax.set_xlabel('Threads', fontsize=LABEL_FONTSIZE)
    ax.set_ylabel('Eficiencia (Speedup / Threads)', fontsize=LABEL_FONTSIZE)
    ax.legend(fontsize=LEGEND_FONTSIZE)
    ax.tick_params(labelsize=TICK_FONTSIZE)
    ax.grid(True, alpha=0.3)
    save_fig(fig, f'{OUTPUT_DIR}/benchmark_eficiencia_por_N_dynamic_simdON.png')

    # --- Comparacao schedule (N maior, simd ON) ---
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for sched in ['static', 'dynamic', 'guided']:
        data = par[(par['N'] == n_max) & (par['simd'] == 'ON') & (par['schedule'] == sched)].sort_values('threads')
        if not data.empty:
            ax.plot(data['threads'], data['speedup'], 'o-', label=sched)
    ax.plot([1, max_t], [1, max_t], 'k--', alpha=0.3, label='Ideal')
    ax.set_xlabel('Threads', fontsize=LABEL_FONTSIZE)
    ax.set_ylabel('Speedup', fontsize=LABEL_FONTSIZE)
    ax.legend(fontsize=LEGEND_FONTSIZE)
    ax.tick_params(labelsize=TICK_FONTSIZE)
    ax.grid(True, alpha=0.3)
    save_fig(fig, f'{OUTPUT_DIR}/benchmark_speedup_schedule_compare_N{n_max}_simdON.png')

    # --- SIMD ON vs OFF (N maior, dynamic) ---
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for simd_val in ['ON', 'OFF']:
        data = par[(par['N'] == n_max) & (par['schedule'] == 'dynamic') & (par['simd'] == simd_val)].sort_values('threads')
        if not data.empty:
            ax.plot(data['threads'], data['speedup'], 'o-', label=f'SIMD {simd_val}')
    ax.plot([1, max_t], [1, max_t], 'k--', alpha=0.3, label='Ideal')
    ax.set_xlabel('Threads', fontsize=LABEL_FONTSIZE)
    ax.set_ylabel('Speedup', fontsize=LABEL_FONTSIZE)
    ax.legend(fontsize=LEGEND_FONTSIZE)
    ax.tick_params(labelsize=TICK_FONTSIZE)
    ax.grid(True, alpha=0.3)
    save_fig(fig, f'{OUTPUT_DIR}/benchmark_speedup_simd_compare_N{n_max}_dynamic.png')

    # --- Tempo barras (dynamic simd ON, N maior) ---
    fig, ax = plt.subplots(figsize=FIGSIZE_BAR)
    seq_time = seq[seq['N'] == n_max]['tempo_seq'].values[0]
    relevant = par[(par['N'] == n_max) & (par['schedule'] == 'dynamic') & (par['simd'] == 'ON')].sort_values('threads')
    labels_bar = ['SEQ (1T)'] + [f'{int(r.threads)}T' for _, r in relevant.iterrows()]
    times = [seq_time] + list(relevant['tempo_seg'])
    colors = [SEQ_COLOR] + ['#1f77b4'] * len(relevant)
    bars = ax.barh(labels_bar, times, color=colors)
    ax.bar_label(bars, fmt='%.4fs', padding=3, fontsize=TICK_FONTSIZE)
    ax.set_xlabel('Tempo (segundos)', fontsize=LABEL_FONTSIZE)
    ax.tick_params(labelsize=TICK_FONTSIZE)
    ax.grid(True, alpha=0.3, axis='x')
    save_fig(fig, f'{OUTPUT_DIR}/benchmark_tempo_barras_N{n_max}_dynamic_simdON.png')

    # --- Sobreposicao por N ---
    for n_val in sorted(par['N'].unique()):
        subset_n = par[par['N'] == n_val]

        # Speedup
        fig, ax = plt.subplots(figsize=FIGSIZE)
        for variante in sorted(subset_n['variante'].unique()):
            data = subset_n[subset_n['variante'] == variante].sort_values('threads')
            s = get_style(variante)
            ax.plot(data['threads'], data['speedup'], marker=s['marker'], color=s['color'],
                    linestyle=s['linestyle'], linewidth=2, markersize=9, label=s['label'])
        ax.plot([1, subset_n['threads'].max()], [1, subset_n['threads'].max()], 'k--', alpha=0.3, linewidth=1, label='Ideal')
        ax.set_xlabel('Threads', fontsize=LABEL_FONTSIZE)
        ax.set_ylabel('Speedup', fontsize=LABEL_FONTSIZE)
        ax.legend(fontsize=LEGEND_FONTSIZE, loc='upper left')
        ax.tick_params(labelsize=TICK_FONTSIZE)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(sorted(subset_n['threads'].unique()))
        save_fig(fig, f'{OUTPUT_DIR}/benchmark_sobreposicao_speedup_N{n_val}_todas_variantes.png')

        # Eficiencia
        fig, ax = plt.subplots(figsize=FIGSIZE)
        for variante in sorted(subset_n['variante'].unique()):
            data = subset_n[subset_n['variante'] == variante].sort_values('threads')
            s = get_style(variante)
            ax.plot(data['threads'], data['eficiencia'], marker=s['marker'], color=s['color'],
                    linestyle=s['linestyle'], linewidth=2, markersize=9, label=s['label'])
        ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.3, linewidth=1, label='Ideal')
        ax.set_xlabel('Threads', fontsize=LABEL_FONTSIZE)
        ax.set_ylabel('Eficiencia (Speedup / Threads)', fontsize=LABEL_FONTSIZE)
        ax.legend(fontsize=LEGEND_FONTSIZE, loc='upper right')
        ax.tick_params(labelsize=TICK_FONTSIZE)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(sorted(subset_n['threads'].unique()))
        save_fig(fig, f'{OUTPUT_DIR}/benchmark_sobreposicao_eficiencia_N{n_val}_todas_variantes.png')

        # Tempo
        fig, ax = plt.subplots(figsize=FIGSIZE)
        seq_time = seq[seq['N'] == n_val]['tempo_seq'].values[0]
        ax.axhline(y=seq_time, color=SEQ_COLOR, linestyle='-', linewidth=2, label=f'Sequencial ({seq_time:.4f}s)')
        for variante in sorted(subset_n['variante'].unique()):
            data = subset_n[subset_n['variante'] == variante].sort_values('threads')
            s = get_style(variante)
            ax.plot(data['threads'], data['tempo_seg'], marker=s['marker'], color=s['color'],
                    linestyle=s['linestyle'], linewidth=2, markersize=9, label=s['label'])
        ax.set_xlabel('Threads', fontsize=LABEL_FONTSIZE)
        ax.set_ylabel('Tempo (segundos)', fontsize=LABEL_FONTSIZE)
        ax.legend(fontsize=LEGEND_FONTSIZE, loc='upper right')
        ax.tick_params(labelsize=TICK_FONTSIZE)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(sorted(subset_n['threads'].unique()))
        save_fig(fig, f'{OUTPUT_DIR}/benchmark_sobreposicao_tempo_N{n_val}_todas_variantes.png')

    # --- Sobreposicao todos N (dynamic simd ON) ---
    fig, ax = plt.subplots(figsize=FIGSIZE)
    n_colors = {5000: '#1f77b4', 10000: '#ff7f0e', 20000: '#2ca02c', 40000: '#d62728', 50000: '#9467bd'}
    subset_dn = par[(par['schedule'] == 'dynamic') & (par['simd'] == 'ON')]
    for n_val in sorted(subset_dn['N'].unique()):
        data = subset_dn[subset_dn['N'] == n_val].sort_values('threads')
        ax.plot(data['threads'], data['speedup'], 'o-', color=n_colors.get(n_val, '#333'),
                linewidth=2, markersize=9, label=f'N={n_val}')
    ax.plot([1, max_t], [1, max_t], 'k--', alpha=0.3, linewidth=1, label='Ideal')
    ax.set_xlabel('Threads', fontsize=LABEL_FONTSIZE)
    ax.set_ylabel('Speedup', fontsize=LABEL_FONTSIZE)
    ax.legend(fontsize=LEGEND_FONTSIZE)
    ax.tick_params(labelsize=TICK_FONTSIZE)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(sorted(par['threads'].unique()))
    save_fig(fig, f'{OUTPUT_DIR}/benchmark_sobreposicao_speedup_todos_N_dynamic_simdON.png')

    # --- Razao SIMD ON / SIMD OFF por schedule (por N) ---
    sched_colors = {'static': '#1f77b4', 'dynamic': '#2ca02c', 'guided': '#ff7f0e'}
    sched_markers = {'static': 'o', 'dynamic': '^', 'guided': 'D'}
    sched_lines = {'static': '-', 'dynamic': '-', 'guided': '-'}

    for n_val in sorted(par['N'].unique()):
        fig, ax = plt.subplots(figsize=FIGSIZE)

        for sched in ['static', 'dynamic', 'guided']:
            on = par[(par['N'] == n_val) & (par['schedule'] == sched) & (par['simd'] == 'ON')][['threads', 'tempo_seg']].sort_values('threads')
            off = par[(par['N'] == n_val) & (par['schedule'] == sched) & (par['simd'] == 'OFF')][['threads', 'tempo_seg']].sort_values('threads')

            if on.empty or off.empty:
                continue

            merged = on.merge(off, on='threads', suffixes=('_on', '_off'))
            merged['razao'] = merged['tempo_seg_on'] / merged['tempo_seg_off']

            ax.plot(merged['threads'], merged['razao'],
                    marker=sched_markers[sched], color=sched_colors[sched],
                    linestyle=sched_lines[sched], linewidth=2, markersize=9,
                    label=sched)

        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1, label='Sem diferenca (1.0)')
        ax.set_xlabel('Threads', fontsize=LABEL_FONTSIZE)
        ax.set_ylabel('Razao Tempo (SIMD ON / SIMD OFF)', fontsize=LABEL_FONTSIZE)
        ax.legend(fontsize=LEGEND_FONTSIZE)
        ax.tick_params(labelsize=TICK_FONTSIZE)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(sorted(par['threads'].unique()))
        save_fig(fig, f'{OUTPUT_DIR}/benchmark_razao_simd_on_off_N{n_val}.png')

    print()

# ===========================================================================
# 2. Graficos de VTune (barras)
# ===========================================================================
summaries = {}
if VTUNE_DIR and os.path.isdir(VTUNE_DIR):
    for f in sorted(os.listdir(VTUNE_DIR)):
        if f.startswith('summary-') and f.endswith('.csv'):
            tag = f.replace('summary-', '').replace('.csv', '')
            summaries[tag] = parse_vtune_summary(os.path.join(VTUNE_DIR, f))

    if summaries:
        print(f"VTune: {len(summaries)} summaries carregados")

        def sort_key(tag):
            info = extract_tag_info(tag)
            return (0 if tag.startswith('seq') else 1, info['threads'], tag)
        tags = sorted(summaries.keys(), key=sort_key)

        labels = []
        for t in tags:
            info = extract_tag_info(t)
            if t.startswith('seq'):
                labels.append('Sequencial')
            else:
                labels.append(f"{info['schedule']} simd={info['simd']} {info['threads']}T")

        def vtune_bar(metric_name, xlabel, fmt, fname, higher_better=False):
            vals = [(t, l, summaries[t].get(metric_name, 0)) for t, l in zip(tags, labels) if metric_name in summaries[t]]
            if not vals:
                return
            fig, ax = plt.subplots(figsize=FIGSIZE_BAR)
            colors = [SEQ_COLOR if t.startswith('seq') else '#1f77b4' for t, _, _ in vals]
            bars = ax.barh([l for _, l, _ in vals], [v for _, _, v in vals], color=colors)
            ax.bar_label(bars, fmt=fmt, padding=3, fontsize=TICK_FONTSIZE)
            ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE)
            ax.tick_params(labelsize=TICK_FONTSIZE)
            ax.grid(True, alpha=0.3, axis='x')
            save_fig(fig, f'{OUTPUT_DIR}/{fname}')

        vtune_bar('Elapsed Time', 'Elapsed Time (s)', '%.2fs', 'vtune_barras_elapsed_time.png')
        vtune_bar('CPI Rate', 'CPI (Cycles Per Instruction)', '%.3f', 'vtune_barras_cpi_rate.png')
        vtune_bar('Microarchitecture Usage', 'Microarchitecture Usage (%)', '%.1f%%', 'vtune_barras_microarch_usage.png')
        vtune_bar('CPU Time', 'CPU Time total (s)', '%.2fs', 'vtune_barras_cpu_time_total.png')

        # Instructions (em bilhoes)
        instr = [(t, l, summaries[t].get('Instructions Retired', 0) / 1e9) for t, l in zip(tags, labels) if 'Instructions Retired' in summaries[t]]
        if instr:
            fig, ax = plt.subplots(figsize=FIGSIZE_BAR)
            colors = [SEQ_COLOR if t.startswith('seq') else '#1f77b4' for t, _, _ in instr]
            bars = ax.barh([l for _, l, _ in instr], [v for _, _, v in instr], color=colors)
            ax.bar_label(bars, fmt='%.1fB', padding=3, fontsize=TICK_FONTSIZE)
            ax.set_xlabel('Instructions Retired (bilhoes)', fontsize=LABEL_FONTSIZE)
            ax.tick_params(labelsize=TICK_FONTSIZE)
            ax.grid(True, alpha=0.3, axis='x')
            save_fig(fig, f'{OUTPUT_DIR}/vtune_barras_instructions_retired.png')

        # Speedup barras
        seq_elapsed = summaries[tags[0]].get('Elapsed Time', 1) if tags[0].startswith('seq') else None
        if seq_elapsed:
            par_tags = [t for t in tags if not t.startswith('seq')]
            par_labels = [labels[tags.index(t)] for t in par_tags]
            speedups = [seq_elapsed / summaries[t].get('Elapsed Time', 1) for t in par_tags]
            fig, ax = plt.subplots(figsize=FIGSIZE_BAR)
            bars = ax.barh(par_labels, speedups, color='#2ca02c')
            ax.bar_label(bars, fmt='%.1fx', padding=3, fontsize=TICK_FONTSIZE)
            ax.set_xlabel('Speedup (vs Sequencial)', fontsize=LABEL_FONTSIZE)
            ax.tick_params(labelsize=TICK_FONTSIZE)
            ax.grid(True, alpha=0.3, axis='x')
            save_fig(fig, f'{OUTPUT_DIR}/vtune_barras_speedup.png')

    # --- Hotspots ---
    hotspot_files = {}
    for f in sorted(os.listdir(VTUNE_DIR)):
        if f.startswith('hotspots-') and f.endswith('.csv') and 'summary' not in f:
            tag = f.replace('hotspots-', '').replace('.csv', '')
            hotspot_files[tag] = os.path.join(VTUNE_DIR, f)

    # --- Hotspots comparativos (publication-ready) ---
    import numpy as np

    # Casos a comparar para cada N
    compare_cases = {
        'seq': 'Sequencial',
        'par_dynamic_simdON_8T': 'dynamic simd=ON 8T',
        'par_dynamic_simdON_20T': 'dynamic simd=ON 20T',
        'par_dynamic_simdON_40T': 'dynamic simd=ON 40T',
        'par_dynamic_simdOFF_8T': 'dynamic simd=OFF 8T',
        'par_static_simdON_8T': 'static simd=ON 8T',
    }

    case_colors = {
        'seq': SEQ_COLOR,
        'par_dynamic_simdON_8T': '#2ca02c',
        'par_dynamic_simdON_20T': '#ff7f0e',
        'par_dynamic_simdON_40T': '#9467bd',
        'par_dynamic_simdOFF_8T': '#e377c2',
        'par_static_simdON_8T': '#1f77b4',
    }

    case_hatches = {
        'seq': '',
        'par_dynamic_simdON_8T': '///',
        'par_dynamic_simdON_20T': '\\\\\\',
        'par_dynamic_simdON_40T': 'xxx',
        'par_dynamic_simdOFF_8T': '...',
        'par_static_simdON_8T': '---',
    }

    for n_val in [5000, 10000, 20000, 40000]:
        # Coletar funcoes para cada caso
        case_data = {}
        for case_prefix, case_label in compare_cases.items():
            tag = f"{case_prefix}_N{n_val}"
            if tag in hotspot_files:
                funcs = parse_vtune_hotspots(hotspot_files[tag])
                if funcs:
                    case_data[case_prefix] = {'label': case_label, 'funcs': {f['Function']: f['CPU Time'] for f in funcs}}

        if len(case_data) < 2:
            continue

        # Pegar top funcoes do sequencial + paralelo
        all_funcs = set()
        for case in case_data.values():
            for func_name in list(case['funcs'].keys())[:5]:
                # Limpar nomes de funcoes do sistema
                if not any(skip in func_name for skip in ['vmlinux', 'copy_user', 'entry_', 'hrtimer', '__calc', 'common_interrupt', 'update_']):
                    all_funcs.add(func_name)

        # Renomear funcoes longas
        func_rename = {
            'knn_sequential': 'knn_sequential',
            'knn_parallel._omp_fn.0': 'knn_parallel (OMP)',
            'calculate_distance': 'calculate_distance',
            '__GI_____strtod_l_internal': 'strtod (CSV parse)',
            'str_to_mpn.part.0.constprop.0': 'str_to_mpn',
            '__strcspn_sse42': 'strcspn',
            '__strtok_r': 'strtok',
            '__strlen_avx2': 'strlen',
            '__memcpy_avx_unaligned_erms': 'memcpy',
            'load_dataset': 'load_dataset',
            '__strspn_sse42': 'strspn',
            'round_and_return': 'round_and_return',
            '__mpn_lshift': 'mpn_lshift',
            'fgets': 'fgets',
        }

        top_funcs = sorted(all_funcs, key=lambda f: max(c['funcs'].get(f, 0) for c in case_data.values()), reverse=True)[:6]
        display_names = [func_rename.get(f, f) for f in top_funcs]

        # Grafico de barras agrupadas com hachuras
        fig, ax = plt.subplots(figsize=(14, 8))
        n_cases = len(case_data)
        bar_height = 0.8 / n_cases
        y_pos = np.arange(len(top_funcs))

        for i, (case_prefix, case_info) in enumerate(case_data.items()):
            values = [case_info['funcs'].get(f, 0) for f in top_funcs]
            bars = ax.barh(y_pos + i * bar_height, values, bar_height * 0.9,
                          color=case_colors.get(case_prefix, '#333'),
                          hatch=case_hatches.get(case_prefix, ''),
                          edgecolor='white', linewidth=0.5,
                          label=case_info['label'])
            # Labels nos valores > 0
            for bar, val in zip(bars, values):
                if val > 0:
                    ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                           f'{val:.2f}s', va='center', fontsize=TICK_FONTSIZE - 2)

        ax.set_yticks(y_pos + (n_cases - 1) * bar_height / 2)
        ax.set_yticklabels(display_names, fontsize=TICK_FONTSIZE)
        ax.set_xlabel('CPU Time (s)', fontsize=LABEL_FONTSIZE)
        ax.legend(fontsize=LEGEND_FONTSIZE - 2, loc='lower right')
        ax.tick_params(labelsize=TICK_FONTSIZE)
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()
        save_fig(fig, f'{OUTPUT_DIR}/vtune_hotspots_comparativo_N{n_val}.png')

# ===========================================================================
# 3. Graficos VTune sobrepostos (linhas separadas)
# ===========================================================================
if VTUNE_DIR and os.path.isdir(VTUNE_DIR) and summaries:
    vtune_data = []
    for tag in summaries:
        info = extract_tag_info(tag)
        vtune_data.append({
            'tag': tag, 'schedule': info['schedule'], 'simd': info['simd'],
            'threads': info['threads'],
            'elapsed_time': summaries[tag].get('Elapsed Time', 0),
            'cpi': summaries[tag].get('CPI Rate', 0),
            'microarch': summaries[tag].get('Microarchitecture Usage', 0),
            'instructions': summaries[tag].get('Instructions Retired', 0) / 1e9,
            'cpu_time': summaries[tag].get('CPU Time', 0),
        })
    vtune_df = pd.DataFrame(vtune_data)
    seq_row = vtune_df[vtune_df['tag'].str.startswith('seq')]
    par_rows = vtune_df[~vtune_df['tag'].str.startswith('seq')].sort_values('threads')

    if not seq_row.empty and not par_rows.empty:
        seq_elapsed = seq_row['elapsed_time'].values[0]
        par_rows_copy = par_rows.copy()
        par_rows_copy['speedup'] = seq_elapsed / par_rows_copy['elapsed_time']

        # --- Tempo vs Threads (1/2) ---
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.axhline(y=seq_elapsed, color=SEQ_COLOR, linewidth=2, linestyle='-', label=f'Sequencial ({seq_elapsed:.2f}s)')
        plot_vtune_lines(ax, par_rows, 'elapsed_time')
        ax.set_xlabel('Threads', fontsize=LABEL_FONTSIZE)
        ax.set_ylabel('Elapsed Time (s)', fontsize=LABEL_FONTSIZE)
        ax.legend(fontsize=LEGEND_FONTSIZE)
        ax.tick_params(labelsize=TICK_FONTSIZE)
        ax.grid(True, alpha=0.3)
        save_fig(fig, f'{OUTPUT_DIR}/vtune_sobreposicao_tempo_vs_threads_1de2.png')

        # --- Speedup vs Threads (2/2) ---
        fig, ax = plt.subplots(figsize=FIGSIZE)
        plot_vtune_lines(ax, par_rows_copy, 'speedup')
        max_t = par_rows_copy['threads'].max()
        ax.plot([1, max_t], [1, max_t], 'k--', alpha=0.3, linewidth=1, label='Ideal')
        ax.set_xlabel('Threads', fontsize=LABEL_FONTSIZE)
        ax.set_ylabel('Speedup', fontsize=LABEL_FONTSIZE)
        ax.legend(fontsize=LEGEND_FONTSIZE)
        ax.tick_params(labelsize=TICK_FONTSIZE)
        ax.grid(True, alpha=0.3)
        save_fig(fig, f'{OUTPUT_DIR}/vtune_sobreposicao_speedup_vs_threads_2de2.png')

        # --- CPI vs Threads (1/2) ---
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.axhline(y=seq_row['cpi'].values[0], color=SEQ_COLOR, linewidth=2, linestyle='-',
                    label=f'Sequencial (CPI={seq_row["cpi"].values[0]:.3f})')
        plot_vtune_lines(ax, par_rows, 'cpi')
        ax.set_xlabel('Threads', fontsize=LABEL_FONTSIZE)
        ax.set_ylabel('CPI (Cycles Per Instruction)', fontsize=LABEL_FONTSIZE)
        ax.legend(fontsize=LEGEND_FONTSIZE)
        ax.tick_params(labelsize=TICK_FONTSIZE)
        ax.grid(True, alpha=0.3)
        save_fig(fig, f'{OUTPUT_DIR}/vtune_sobreposicao_cpi_vs_threads_1de2.png')

        # --- Microarch vs Threads (2/2) ---
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.axhline(y=seq_row['microarch'].values[0], color=SEQ_COLOR, linewidth=2, linestyle='-',
                    label=f'Sequencial ({seq_row["microarch"].values[0]:.1f}%)')
        plot_vtune_lines(ax, par_rows, 'microarch')
        ax.set_xlabel('Threads', fontsize=LABEL_FONTSIZE)
        ax.set_ylabel('Microarchitecture Usage (%)', fontsize=LABEL_FONTSIZE)
        ax.legend(fontsize=LEGEND_FONTSIZE)
        ax.tick_params(labelsize=TICK_FONTSIZE)
        ax.grid(True, alpha=0.3)
        save_fig(fig, f'{OUTPUT_DIR}/vtune_sobreposicao_microarch_vs_threads_2de2.png')

# ===========================================================================
# 4. Graficos de Memoria (hpc-summary + perf-summary)
# ===========================================================================
if VTUNE_DIR and os.path.isdir(VTUNE_DIR):
    # Carregar hpc-summary e perf-summary
    hpc_summaries = {}
    perf_summaries = {}
    for f in sorted(os.listdir(VTUNE_DIR)):
        if f.startswith('hpc-summary-') and f.endswith('.csv'):
            tag = f.replace('hpc-summary-', '').replace('.csv', '')
            hpc_summaries[tag] = parse_vtune_summary(os.path.join(VTUNE_DIR, f))
        elif f.startswith('perf-summary-') and f.endswith('.csv'):
            tag = f.replace('perf-summary-', '').replace('.csv', '')
            perf_summaries[tag] = parse_vtune_summary(os.path.join(VTUNE_DIR, f))

    # Usar APENAS hpc-summary para metricas de memoria (mais confiavel)
    # e perf-summary para NUMA (que so aparece no perf)
    if hpc_summaries:
        print(f"\nMemoria: {len(hpc_summaries)} hpc-summaries + {len(perf_summaries)} perf-summaries")

        all_mem_data = []
        for tag, metrics in hpc_summaries.items():
            info = extract_tag_info(tag)
            # Buscar NUMA no perf-summary correspondente
            numa_val = 0
            if tag in perf_summaries:
                numa_raw = perf_summaries[tag].get('NUMA: % of Remote Accesses', 0)
                if isinstance(numa_raw, (int, float)):
                    numa_val = float(numa_raw)

            entry = {
                'tag': tag, 'schedule': info['schedule'], 'simd': info['simd'],
                'threads': info['threads'],
                'memory_bound': metrics.get('Memory Bound', 0),
                'cache_bound': metrics.get('Cache Bound', 0),
                'cpi': metrics.get('CPI Rate', 0),
                'dram_bw_bound': metrics.get('DRAM Bandwidth Bound', 0),
                'numa_remote': numa_val,
            }
            for k in ['memory_bound', 'cache_bound', 'cpi', 'dram_bw_bound', 'numa_remote']:
                if isinstance(entry[k], str):
                    m = re.match(r'([\d.]+)', str(entry[k]))
                    entry[k] = float(m.group(1)) if m else 0
            all_mem_data.append(entry)

        # Adicionar seq do perf-summary (hpc nao tem seq)
        for tag, metrics in perf_summaries.items():
            if tag.startswith('seq'):
                info = extract_tag_info(tag)
                numa_raw = metrics.get('NUMA: % of Remote Accesses', 0)
                if isinstance(numa_raw, str):
                    m_n = re.match(r'([\d.]+)', str(numa_raw))
                    numa_raw = float(m_n.group(1)) if m_n else 0
                entry = {
                    'tag': tag, 'schedule': info['schedule'], 'simd': info['simd'],
                    'threads': info['threads'],
                    'memory_bound': metrics.get('Memory Bound', 0),
                    'cache_bound': metrics.get('Cache Bound', metrics.get('L1 Bound', 0)),
                    'cpi': metrics.get('CPI Rate', 0),
                    'dram_bw_bound': metrics.get('DRAM Bandwidth Bound', 0),
                    'numa_remote': numa_raw,
                }
                for k in ['memory_bound', 'cache_bound', 'cpi', 'dram_bw_bound', 'numa_remote']:
                    if isinstance(entry[k], str):
                        m_v = re.match(r'([\d.]+)', str(entry[k]))
                        entry[k] = float(m_v.group(1)) if m_v else 0
                all_mem_data.append(entry)

        mem_df = pd.DataFrame(all_mem_data)
        seq_mem = mem_df[mem_df['tag'].str.startswith('seq')]
        par_mem = mem_df[~mem_df['tag'].str.startswith('seq')]

        # Filtrar por N=40000 para graficos de linha (mais dados confiaveis)
        par_mem_40k = par_mem[par_mem['tag'].str.contains('N40000')]

        if not par_mem_40k.empty:
            # --- Memory Bound vs Threads ---
            fig, ax = plt.subplots(figsize=FIGSIZE)
            seq_val = seq_mem[seq_mem['tag'].str.contains('N40000')]['memory_bound'].values
            if len(seq_val) > 0 and seq_val[0] > 0:
                ax.axhline(y=seq_val[0], color=SEQ_COLOR, linewidth=2, linestyle='-',
                            label=f'Sequencial ({seq_val[0]:.1f}%)')
            plot_vtune_lines(ax, par_mem_40k, 'memory_bound')
            ax.set_xlabel('Threads', fontsize=LABEL_FONTSIZE)
            ax.set_ylabel('Memory Bound (%)', fontsize=LABEL_FONTSIZE)
            ax.legend(fontsize=LEGEND_FONTSIZE)
            ax.tick_params(labelsize=TICK_FONTSIZE)
            ax.grid(True, alpha=0.3)
            save_fig(fig, f'{OUTPUT_DIR}/vtune_sobreposicao_memory_bound_vs_threads_N40000.png')

            # --- Cache Bound vs Threads ---
            fig, ax = plt.subplots(figsize=FIGSIZE)
            seq_val = seq_mem[seq_mem['tag'].str.contains('N40000')]['cache_bound'].values
            if len(seq_val) > 0 and seq_val[0] > 0:
                ax.axhline(y=seq_val[0], color=SEQ_COLOR, linewidth=2, linestyle='-',
                            label=f'Sequencial ({seq_val[0]:.1f}%)')
            plot_vtune_lines(ax, par_mem_40k, 'cache_bound')
            ax.set_xlabel('Threads', fontsize=LABEL_FONTSIZE)
            ax.set_ylabel('Cache Bound (%)', fontsize=LABEL_FONTSIZE)
            ax.legend(fontsize=LEGEND_FONTSIZE)
            ax.tick_params(labelsize=TICK_FONTSIZE)
            ax.grid(True, alpha=0.3)
            save_fig(fig, f'{OUTPUT_DIR}/vtune_sobreposicao_cache_bound_vs_threads_N40000.png')

            # --- NUMA Remote Accesses vs Threads ---
            numa_data = par_mem_40k[par_mem_40k['numa_remote'] > 0]
            seq_numa = seq_mem[seq_mem['tag'].str.contains('N40000')]['numa_remote'].values
            fig, ax = plt.subplots(figsize=FIGSIZE)
            if len(seq_numa) > 0 and seq_numa[0] > 0:
                ax.axhline(y=seq_numa[0], color=SEQ_COLOR, linewidth=2, linestyle='-',
                            label=f'Sequencial ({seq_numa[0]:.1f}%)')
            plot_vtune_lines(ax, par_mem_40k, 'numa_remote')
            ax.set_xlabel('Threads', fontsize=LABEL_FONTSIZE)
            ax.set_ylabel('NUMA Remote Accesses (%)', fontsize=LABEL_FONTSIZE)
            ax.legend(fontsize=LEGEND_FONTSIZE)
            ax.tick_params(labelsize=TICK_FONTSIZE)
            ax.grid(True, alpha=0.3)
            save_fig(fig, f'{OUTPUT_DIR}/vtune_sobreposicao_numa_remote_vs_threads_N40000.png')

        # --- Barras: Memory Bound por caso (N=40000) ---
        mem_40k_all = mem_df[mem_df['tag'].str.contains('N40000')].copy()
        if not mem_40k_all.empty:
            mem_40k_all = mem_40k_all.sort_values('threads')
            bar_labels = []
            for _, row in mem_40k_all.iterrows():
                if row['tag'].startswith('seq'):
                    bar_labels.append('Sequencial')
                else:
                    bar_labels.append(f"{row['schedule']} simd={row['simd']} {int(row['threads'])}T")
            mem_40k_all['bar_label'] = bar_labels

            # Memory Bound barras
            fig, ax = plt.subplots(figsize=FIGSIZE_BAR)
            colors = [SEQ_COLOR if t.startswith('seq') else '#1f77b4' for t in mem_40k_all['tag']]
            bars = ax.barh(mem_40k_all['bar_label'], mem_40k_all['memory_bound'], color=colors)
            ax.bar_label(bars, fmt='%.1f%%', padding=3, fontsize=TICK_FONTSIZE)
            ax.set_xlabel('Memory Bound (%)', fontsize=LABEL_FONTSIZE)
            ax.tick_params(labelsize=TICK_FONTSIZE)
            ax.grid(True, alpha=0.3, axis='x')
            save_fig(fig, f'{OUTPUT_DIR}/vtune_barras_memory_bound_N40000.png')

            # Cache Bound barras
            fig, ax = plt.subplots(figsize=FIGSIZE_BAR)
            bars = ax.barh(mem_40k_all['bar_label'], mem_40k_all['cache_bound'], color=colors)
            ax.bar_label(bars, fmt='%.1f%%', padding=3, fontsize=TICK_FONTSIZE)
            ax.set_xlabel('Cache Bound (%)', fontsize=LABEL_FONTSIZE)
            ax.tick_params(labelsize=TICK_FONTSIZE)
            ax.grid(True, alpha=0.3, axis='x')
            save_fig(fig, f'{OUTPUT_DIR}/vtune_barras_cache_bound_N40000.png')

print()
print(f"Graficos salvos em: {OUTPUT_DIR}/")
print(f"Total: {len(os.listdir(OUTPUT_DIR))} arquivos")
