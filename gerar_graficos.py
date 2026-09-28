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

# Auto-detectar CSV
if CSV_BENCHMARK is None:
    for f in ["resultados_knn_sem_vtune.csv", "resultados_knn.csv"]:
        if os.path.exists(f):
            CSV_BENCHMARK = f
            break

# Auto-detectar VTune reports
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


def parse_vtune_summary(filepath):
    """Parse um summary CSV do VTune e retorna dict de metricas."""
    metrics = {}
    with open(filepath, 'r') as fh:
        for line in fh:
            if '\t' in line and 'vtune:' not in line:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    name = parts[1].strip()
                    val_str = parts[2].strip()
                    # Tentar extrair numero
                    try:
                        metrics[name] = float(val_str)
                    except ValueError:
                        # Tentar extrair percentual: "34.7% (6.943 out of 20)"
                        m = re.match(r'([\d.]+)%', val_str)
                        if m:
                            metrics[name] = float(m.group(1))
                        else:
                            metrics[name] = val_str
    return metrics


def parse_vtune_hotspots(filepath):
    """Parse um hotspots CSV do VTune e retorna lista de dicts."""
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
    """Extrai informacoes do tag (ex: dynamic_simd_8T -> schedule, simd, threads)."""
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
    label = f"{schedule}_{simd}_{threads}T"
    return {'schedule': schedule, 'simd': simd, 'threads': threads, 'label': label}


# ---------------------------------------------------------------------------
# 1. Graficos de Benchmark CSV (se disponivel)
# ---------------------------------------------------------------------------
if CSV_BENCHMARK and os.path.exists(CSV_BENCHMARK):
    df = pd.read_csv(CSV_BENCHMARK)
    print(f"Benchmark: {len(df)} linhas carregadas")

    seq = df[df['versao'] == 'sequencial'][['N', 'tempo_seg']].rename(columns={'tempo_seg': 'tempo_seq'})
    par = df[df['versao'] == 'paralelo'].copy()
    par = par.merge(seq, on='N')
    par['speedup'] = par['tempo_seq'] / par['tempo_seg']
    par['eficiencia'] = par['speedup'] / par['threads']

    # --- Speedup vs Threads por N (dynamic, simd ON) ---
    fig, ax = plt.subplots(figsize=(10, 6))
    subset = par[(par['schedule'] == 'dynamic') & (par['simd'] == 'ON')]
    for n_val in sorted(subset['N'].unique()):
        data = subset[subset['N'] == n_val].sort_values('threads')
        ax.plot(data['threads'], data['speedup'], 'o-', label=f'N={n_val}')
    max_t = par['threads'].max()
    ax.plot([1, max_t], [1, max_t], 'k--', alpha=0.3, label='Ideal')
    ax.set_xlabel('Threads')
    ax.set_ylabel('Speedup')
    ax.set_title('Speedup vs Threads (dynamic, SIMD ON)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/speedup_por_N.png', dpi=150)
    plt.close(fig)
    print(f"  -> {OUTPUT_DIR}/speedup_por_N.png")

    # --- Eficiencia vs Threads por N ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for n_val in sorted(subset['N'].unique()):
        data = subset[subset['N'] == n_val].sort_values('threads')
        ax.plot(data['threads'], data['eficiencia'], 'o-', label=f'N={n_val}')
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.3, label='Ideal')
    ax.set_xlabel('Threads')
    ax.set_ylabel('Eficiencia')
    ax.set_title('Eficiencia vs Threads (dynamic, SIMD ON)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/eficiencia_por_N.png', dpi=150)
    plt.close(fig)
    print(f"  -> {OUTPUT_DIR}/eficiencia_por_N.png")

    # --- Comparacao schedule (N maior, simd ON) ---
    n_max = par['N'].max()
    fig, ax = plt.subplots(figsize=(10, 6))
    for sched in ['static', 'dynamic', 'guided']:
        data = par[(par['N'] == n_max) & (par['simd'] == 'ON') & (par['schedule'] == sched)].sort_values('threads')
        if not data.empty:
            ax.plot(data['threads'], data['speedup'], 'o-', label=sched)
    ax.plot([1, max_t], [1, max_t], 'k--', alpha=0.3, label='Ideal')
    ax.set_xlabel('Threads')
    ax.set_ylabel('Speedup')
    ax.set_title(f'static vs dynamic vs guided (N={n_max}, SIMD ON)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/speedup_schedule_compare.png', dpi=150)
    plt.close(fig)
    print(f"  -> {OUTPUT_DIR}/speedup_schedule_compare.png")

    # --- SIMD ON vs OFF (N maior, dynamic) ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for simd_val in ['ON', 'OFF']:
        data = par[(par['N'] == n_max) & (par['schedule'] == 'dynamic') & (par['simd'] == simd_val)].sort_values('threads')
        if not data.empty:
            ax.plot(data['threads'], data['speedup'], 'o-', label=f'SIMD {simd_val}')
    ax.plot([1, max_t], [1, max_t], 'k--', alpha=0.3, label='Ideal')
    ax.set_xlabel('Threads')
    ax.set_ylabel('Speedup')
    ax.set_title(f'SIMD ON vs OFF (N={n_max}, dynamic)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/speedup_simd_compare.png', dpi=150)
    plt.close(fig)
    print(f"  -> {OUTPUT_DIR}/speedup_simd_compare.png")

    # --- Tempo absoluto (barras, dynamic simd ON) ---
    fig, ax = plt.subplots(figsize=(12, 6))
    seq_time = seq[seq['N'] == n_max]['tempo_seq'].values[0]
    relevant = par[(par['N'] == n_max) & (par['schedule'] == 'dynamic') & (par['simd'] == 'ON')].sort_values('threads')
    labels = ['SEQ (1T)'] + [f'{int(r.threads)}T' for _, r in relevant.iterrows()]
    times = [seq_time] + list(relevant['tempo_seg'])
    colors = ['#d62728'] + ['#1f77b4'] * len(relevant)
    ax.barh(labels, times, color=colors)
    ax.set_xlabel('Tempo (segundos)')
    ax.set_title(f'Tempo de execucao (N={n_max}, dynamic, SIMD ON)')
    ax.grid(True, alpha=0.3, axis='x')
    fig.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/tempo_barras.png', dpi=150)
    plt.close(fig)
    print(f"  -> {OUTPUT_DIR}/tempo_barras.png")
    print()

# ---------------------------------------------------------------------------
# 2. Graficos de VTune
# ---------------------------------------------------------------------------
if VTUNE_DIR and os.path.isdir(VTUNE_DIR):
    # Carregar todos os summaries
    summaries = {}
    for f in sorted(os.listdir(VTUNE_DIR)):
        if f.startswith('summary-') and f.endswith('.csv'):
            tag = f.replace('summary-', '').replace('.csv', '')
            summaries[tag] = parse_vtune_summary(os.path.join(VTUNE_DIR, f))

    if summaries:
        print(f"VTune: {len(summaries)} summaries carregados")

        # Ordenar tags: seq primeiro, depois por threads
        def sort_key(tag):
            info = extract_tag_info(tag)
            return (0 if tag.startswith('seq') else 1, info['threads'], tag)
        tags = sorted(summaries.keys(), key=sort_key)

        # Labels legiveis
        labels = []
        for t in tags:
            info = extract_tag_info(t)
            if t.startswith('seq'):
                labels.append('Sequencial')
            else:
                labels.append(f"{info['schedule']} simd={info['simd']} {info['threads']}T")

        # --- Elapsed Time ---
        elapsed = [summaries[t].get('Elapsed Time', 0) for t in tags]
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#d62728' if t.startswith('seq') else '#1f77b4' for t in tags]
        bars = ax.barh(labels, elapsed, color=colors)
        ax.bar_label(bars, fmt='%.2fs', padding=3)
        ax.set_xlabel('Elapsed Time (s)')
        ax.set_title('VTune: Tempo de Execucao (N=40000)')
        ax.grid(True, alpha=0.3, axis='x')
        fig.tight_layout()
        fig.savefig(f'{OUTPUT_DIR}/vtune_elapsed_time.png', dpi=150)
        plt.close(fig)
        print(f"  -> {OUTPUT_DIR}/vtune_elapsed_time.png")

        # --- Speedup calculado do VTune ---
        seq_elapsed = summaries[tags[0]].get('Elapsed Time', 1) if tags[0].startswith('seq') else None
        if seq_elapsed:
            par_tags = [t for t in tags if not t.startswith('seq')]
            par_labels = [labels[tags.index(t)] for t in par_tags]
            speedups = [seq_elapsed / summaries[t].get('Elapsed Time', 1) for t in par_tags]
            threads_list = [extract_tag_info(t)['threads'] for t in par_tags]

            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(par_labels, speedups, color='#2ca02c')
            ax.bar_label(bars, fmt='%.1fx', padding=3)
            ax.set_xlabel('Speedup (vs Sequencial)')
            ax.set_title(f'VTune: Speedup (N=40000, T_seq={seq_elapsed:.2f}s)')
            ax.grid(True, alpha=0.3, axis='x')
            fig.tight_layout()
            fig.savefig(f'{OUTPUT_DIR}/vtune_speedup.png', dpi=150)
            plt.close(fig)
            print(f"  -> {OUTPUT_DIR}/vtune_speedup.png")

        # --- CPI Rate ---
        cpi = [(t, l, summaries[t].get('CPI Rate', 0)) for t, l in zip(tags, labels) if 'CPI Rate' in summaries[t]]
        if cpi:
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['#d62728' if t.startswith('seq') else '#1f77b4' for t, _, _ in cpi]
            bars = ax.barh([l for _, l, _ in cpi], [v for _, _, v in cpi], color=colors)
            ax.bar_label(bars, fmt='%.3f', padding=3)
            ax.set_xlabel('CPI (Cycles Per Instruction)')
            ax.set_title('VTune: CPI Rate (menor = melhor)')
            ax.grid(True, alpha=0.3, axis='x')
            fig.tight_layout()
            fig.savefig(f'{OUTPUT_DIR}/vtune_cpi_rate.png', dpi=150)
            plt.close(fig)
            print(f"  -> {OUTPUT_DIR}/vtune_cpi_rate.png")

        # --- Microarchitecture Usage ---
        uarch = [(t, l, summaries[t].get('Microarchitecture Usage', 0)) for t, l in zip(tags, labels) if 'Microarchitecture Usage' in summaries[t]]
        if uarch:
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['#d62728' if t.startswith('seq') else '#1f77b4' for t, _, _ in uarch]
            bars = ax.barh([l for _, l, _ in uarch], [v for _, _, v in uarch], color=colors)
            ax.bar_label(bars, fmt='%.1f%%', padding=3)
            ax.set_xlabel('Microarchitecture Usage (%)')
            ax.set_title('VTune: Uso de Microarquitetura (maior = melhor)')
            ax.grid(True, alpha=0.3, axis='x')
            fig.tight_layout()
            fig.savefig(f'{OUTPUT_DIR}/vtune_microarch_usage.png', dpi=150)
            plt.close(fig)
            print(f"  -> {OUTPUT_DIR}/vtune_microarch_usage.png")

        # --- Instructions Retired ---
        instr = [(t, l, summaries[t].get('Instructions Retired', 0) / 1e9) for t, l in zip(tags, labels) if 'Instructions Retired' in summaries[t]]
        if instr:
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['#d62728' if t.startswith('seq') else '#1f77b4' for t, _, _ in instr]
            bars = ax.barh([l for _, l, _ in instr], [v for _, _, v in instr], color=colors)
            ax.bar_label(bars, fmt='%.1fB', padding=3)
            ax.set_xlabel('Instructions Retired (bilhoes)')
            ax.set_title('VTune: Instrucoes Executadas')
            ax.grid(True, alpha=0.3, axis='x')
            fig.tight_layout()
            fig.savefig(f'{OUTPUT_DIR}/vtune_instructions.png', dpi=150)
            plt.close(fig)
            print(f"  -> {OUTPUT_DIR}/vtune_instructions.png")

        # --- CPU Time (total, todas as threads) ---
        cpu_time = [(t, l, summaries[t].get('CPU Time', 0)) for t, l in zip(tags, labels) if 'CPU Time' in summaries[t]]
        if cpu_time:
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['#d62728' if t.startswith('seq') else '#1f77b4' for t, _, _ in cpu_time]
            bars = ax.barh([l for _, l, _ in cpu_time], [v for _, _, v in cpu_time], color=colors)
            ax.bar_label(bars, fmt='%.2fs', padding=3)
            ax.set_xlabel('CPU Time total (s) - soma de todas as threads')
            ax.set_title('VTune: CPU Time Total')
            ax.grid(True, alpha=0.3, axis='x')
            fig.tight_layout()
            fig.savefig(f'{OUTPUT_DIR}/vtune_cpu_time_total.png', dpi=150)
            plt.close(fig)
            print(f"  -> {OUTPUT_DIR}/vtune_cpu_time_total.png")

    # --- Hotspots: sequencial e paralelos ---
    hotspot_files = {}
    for f in sorted(os.listdir(VTUNE_DIR)):
        if f.startswith('hotspots-') and f.endswith('.csv') and 'summary' not in f:
            tag = f.replace('hotspots-', '').replace('.csv', '')
            hotspot_files[tag] = os.path.join(VTUNE_DIR, f)

    for tag, filepath in hotspot_files.items():
        functions = parse_vtune_hotspots(filepath)
        if not functions:
            continue

        info = extract_tag_info(tag)
        if tag.startswith('seq'):
            title = f'Hotspots: Sequencial (N=40000)'
        else:
            title = f'Hotspots: {info["schedule"]} simd={info["simd"]} {info["threads"]}T (N=40000)'

        func_df = pd.DataFrame(functions).head(8)
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(func_df['Function'][::-1], func_df['CPU Time'][::-1])
        ax.bar_label(bars, fmt='%.3fs', padding=3)
        ax.set_xlabel('CPU Time (s)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='x')
        fig.tight_layout()
        fname = f'vtune_hotspots_{tag}.png'
        fig.savefig(f'{OUTPUT_DIR}/{fname}', dpi=150)
        plt.close(fig)
        print(f"  -> {OUTPUT_DIR}/{fname}")

# ---------------------------------------------------------------------------
# 3. Graficos de linha+pontos sobrepostos (benchmark CSV)
# ---------------------------------------------------------------------------
if CSV_BENCHMARK and os.path.exists(CSV_BENCHMARK):
    df = pd.read_csv(CSV_BENCHMARK)
    seq = df[df['versao'] == 'sequencial'][['N', 'tempo_seg']].rename(columns={'tempo_seg': 'tempo_seq'})
    par = df[df['versao'] == 'paralelo'].copy()
    par = par.merge(seq, on='N')
    par['speedup'] = par['tempo_seq'] / par['tempo_seg']
    par['eficiencia'] = par['speedup'] / par['threads']
    par['variante'] = par['schedule'] + ' simd=' + par['simd']

    # Estilos visuais: cor + marker + tipo de linha diferentes para cada variante
    STYLES = {
        'static ON':  {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-',  'label': 'static simd=ON'},
        'static OFF': {'color': '#1f77b4', 'marker': 's', 'linestyle': '--', 'label': 'static simd=OFF'},
        'dynamic ON':  {'color': '#2ca02c', 'marker': '^', 'linestyle': '-',  'label': 'dynamic simd=ON'},
        'dynamic OFF': {'color': '#2ca02c', 'marker': 'v', 'linestyle': '--', 'label': 'dynamic simd=OFF'},
        'guided ON':  {'color': '#ff7f0e', 'marker': 'D', 'linestyle': '-',  'label': 'guided simd=ON'},
        'guided OFF': {'color': '#ff7f0e', 'marker': 'x', 'linestyle': '--', 'label': 'guided simd=OFF'},
    }

    def get_style(variante):
        key = variante.replace('simd=', '')
        return STYLES.get(key, {'color': '#333', 'marker': 'o', 'linestyle': '-', 'label': variante})

    for n_val in sorted(par['N'].unique()):
        subset = par[par['N'] == n_val]

        # --- Speedup sobreposicao: todas as variantes ---
        fig, ax = plt.subplots(figsize=(12, 7))
        for variante in sorted(subset['variante'].unique()):
            data = subset[subset['variante'] == variante].sort_values('threads')
            s = get_style(variante)
            ax.plot(data['threads'], data['speedup'], marker=s['marker'], color=s['color'],
                    linestyle=s['linestyle'], linewidth=2, markersize=8, label=s['label'])
        max_t = subset['threads'].max()
        ax.plot([1, max_t], [1, max_t], 'k--', alpha=0.3, linewidth=1, label='Ideal')
        ax.set_xlabel('Threads', fontsize=12)
        ax.set_ylabel('Speedup', fontsize=12)
        ax.set_title(f'Speedup: Todas as Variantes (N={n_val})', fontsize=14)
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(sorted(subset['threads'].unique()))
        fig.tight_layout()
        fig.savefig(f'{OUTPUT_DIR}/sobreposicao_speedup_N{n_val}.png', dpi=150)
        plt.close(fig)
        print(f"  -> {OUTPUT_DIR}/sobreposicao_speedup_N{n_val}.png")

        # --- Eficiencia sobreposicao ---
        fig, ax = plt.subplots(figsize=(12, 7))
        for variante in sorted(subset['variante'].unique()):
            data = subset[subset['variante'] == variante].sort_values('threads')
            s = get_style(variante)
            ax.plot(data['threads'], data['eficiencia'], marker=s['marker'], color=s['color'],
                    linestyle=s['linestyle'], linewidth=2, markersize=8, label=s['label'])
        ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.3, linewidth=1, label='Ideal')
        ax.set_xlabel('Threads', fontsize=12)
        ax.set_ylabel('Eficiencia (Speedup / Threads)', fontsize=12)
        ax.set_title(f'Eficiencia: Todas as Variantes (N={n_val})', fontsize=14)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(sorted(subset['threads'].unique()))
        fig.tight_layout()
        fig.savefig(f'{OUTPUT_DIR}/sobreposicao_eficiencia_N{n_val}.png', dpi=150)
        plt.close(fig)
        print(f"  -> {OUTPUT_DIR}/sobreposicao_eficiencia_N{n_val}.png")

        # --- Tempo absoluto sobreposicao ---
        fig, ax = plt.subplots(figsize=(12, 7))
        seq_time = seq[seq['N'] == n_val]['tempo_seq'].values[0]
        ax.axhline(y=seq_time, color='#d62728', linestyle='-', linewidth=2, label=f'Sequencial ({seq_time:.4f}s)')
        for variante in sorted(subset['variante'].unique()):
            data = subset[subset['variante'] == variante].sort_values('threads')
            s = get_style(variante)
            ax.plot(data['threads'], data['tempo_seg'], marker=s['marker'], color=s['color'],
                    linestyle=s['linestyle'], linewidth=2, markersize=8, label=s['label'])
        ax.set_xlabel('Threads', fontsize=12)
        ax.set_ylabel('Tempo (segundos)', fontsize=12)
        ax.set_title(f'Tempo de Execucao: Todas as Variantes (N={n_val})', fontsize=14)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(sorted(subset['threads'].unique()))
        fig.tight_layout()
        fig.savefig(f'{OUTPUT_DIR}/sobreposicao_tempo_N{n_val}.png', dpi=150)
        plt.close(fig)
        print(f"  -> {OUTPUT_DIR}/sobreposicao_tempo_N{n_val}.png")

    # --- Sobreposicao de N: speedup por tamanho (dynamic simd ON) ---
    fig, ax = plt.subplots(figsize=(12, 7))
    n_colors = {5000: '#1f77b4', 10000: '#ff7f0e', 20000: '#2ca02c', 40000: '#d62728', 50000: '#9467bd'}
    subset_dn = par[(par['schedule'] == 'dynamic') & (par['simd'] == 'ON')]
    for n_val in sorted(subset_dn['N'].unique()):
        data = subset_dn[subset_dn['N'] == n_val].sort_values('threads')
        ax.plot(data['threads'], data['speedup'], 'o-', color=n_colors.get(n_val, '#333'),
                linewidth=2, markersize=8, label=f'N={n_val}')
    max_t = par['threads'].max()
    ax.plot([1, max_t], [1, max_t], 'k--', alpha=0.3, linewidth=1, label='Ideal')
    ax.set_xlabel('Threads', fontsize=12)
    ax.set_ylabel('Speedup', fontsize=12)
    ax.set_title('Speedup vs Threads por Tamanho (dynamic, SIMD ON)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(sorted(par['threads'].unique()))
    fig.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/sobreposicao_speedup_todos_N.png', dpi=150)
    plt.close(fig)
    print(f"  -> {OUTPUT_DIR}/sobreposicao_speedup_todos_N.png")

# ---------------------------------------------------------------------------
# 4. Graficos de linha+pontos sobrepostos (VTune)
# ---------------------------------------------------------------------------
if VTUNE_DIR and os.path.isdir(VTUNE_DIR) and summaries:
    # Agrupar por variante (schedule+simd) e ordenar por threads
    vtune_data = []
    for tag in summaries:
        info = extract_tag_info(tag)
        vtune_data.append({
            'tag': tag,
            'label': info['label'],
            'schedule': info['schedule'],
            'simd': info['simd'],
            'threads': info['threads'],
            'elapsed_time': summaries[tag].get('Elapsed Time', 0),
            'cpi': summaries[tag].get('CPI Rate', 0),
            'microarch': summaries[tag].get('Microarchitecture Usage', 0),
            'instructions': summaries[tag].get('Instructions Retired', 0) / 1e9,
            'cpu_time': summaries[tag].get('CPU Time', 0),
        })
    vtune_df = pd.DataFrame(vtune_data)

    # Separar seq e par
    seq_row = vtune_df[vtune_df['tag'].str.startswith('seq')]
    par_rows = vtune_df[~vtune_df['tag'].str.startswith('seq')].sort_values('threads')

    if not seq_row.empty and not par_rows.empty:
        seq_elapsed = seq_row['elapsed_time'].values[0]

        # Estilos para variantes VTune (cor + marker + linha)
        VTUNE_STYLES = {
            'dynamic_simd':   {'color': '#2ca02c', 'marker': '^', 'linestyle': '-',  'label': 'dynamic simd=ON'},
            'dynamic_nosimd': {'color': '#2ca02c', 'marker': 'v', 'linestyle': '--', 'label': 'dynamic simd=OFF'},
            'static_simd':    {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-',  'label': 'static simd=ON'},
            'static_nosimd':  {'color': '#1f77b4', 'marker': 's', 'linestyle': '--', 'label': 'static simd=OFF'},
            'guided_simd':    {'color': '#ff7f0e', 'marker': 'D', 'linestyle': '-',  'label': 'guided simd=ON'},
            'guided_nosimd':  {'color': '#ff7f0e', 'marker': 'x', 'linestyle': '--', 'label': 'guided simd=OFF'},
        }

        def get_vtune_variante(tag):
            """Extrai a variante do tag (ex: dynamic_simd de par_dynamic_simd_8T_N40000)."""
            parts = tag.replace('par_', '').split('_')
            # Reconstroi: schedule_simd ou schedule_nosimd
            if 'nosimd' in tag:
                return parts[0] + '_nosimd'
            else:
                return parts[0] + '_simd'

        def plot_vtune_lines(ax, par_data, y_col):
            """Plota linhas com estilos distintos para cada variante."""
            seen = set()
            for _, row in par_data.iterrows():
                var = get_vtune_variante(row['tag'])
                if var not in seen:
                    seen.add(var)
                # Acumular dados por variante
            for var in VTUNE_STYLES:
                mask = par_data['tag'].apply(lambda t: get_vtune_variante(t) == var)
                data = par_data[mask].sort_values('threads')
                if len(data) > 0:
                    s = VTUNE_STYLES[var]
                    ax.plot(data['threads'], data[y_col], marker=s['marker'], color=s['color'],
                            linestyle=s['linestyle'], linewidth=2, markersize=9, label=s['label'])

        # --- VTune: Linha sobreposicao Elapsed Time + Speedup ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Elapsed Time
        ax1.axhline(y=seq_elapsed, color='#d62728', linewidth=2, linestyle='-', label=f'Sequencial ({seq_elapsed:.2f}s)')
        plot_vtune_lines(ax1, par_rows, 'elapsed_time')
        ax1.set_xlabel('Threads', fontsize=12)
        ax1.set_ylabel('Elapsed Time (s)', fontsize=12)
        ax1.set_title('VTune: Tempo vs Threads (N=40000)', fontsize=13)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Speedup
        par_rows_copy = par_rows.copy()
        par_rows_copy['speedup'] = seq_elapsed / par_rows_copy['elapsed_time']
        plot_vtune_lines(ax2, par_rows_copy, 'speedup')
        max_t = par_rows_copy['threads'].max()
        ax2.plot([1, max_t], [1, max_t], 'k--', alpha=0.3, linewidth=1, label='Ideal')
        ax2.set_xlabel('Threads', fontsize=12)
        ax2.set_ylabel('Speedup', fontsize=12)
        ax2.set_title(f'VTune: Speedup vs Threads (T_seq={seq_elapsed:.2f}s)', fontsize=13)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(f'{OUTPUT_DIR}/vtune_sobreposicao_tempo_speedup.png', dpi=150)
        plt.close(fig)
        print(f"  -> {OUTPUT_DIR}/vtune_sobreposicao_tempo_speedup.png")

        # --- VTune: CPI + Microarch sobreposicao ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # CPI por threads
        ax1.axhline(y=seq_row['cpi'].values[0], color='#d62728', linewidth=2, linestyle='-',
                    label=f'Sequencial (CPI={seq_row["cpi"].values[0]:.3f})')
        plot_vtune_lines(ax1, par_rows, 'cpi')
        ax1.set_xlabel('Threads', fontsize=12)
        ax1.set_ylabel('CPI', fontsize=12)
        ax1.set_title('VTune: CPI vs Threads (menor = melhor)', fontsize=13)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Microarch por threads
        ax2.axhline(y=seq_row['microarch'].values[0], color='#d62728', linewidth=2, linestyle='-',
                    label=f'Sequencial ({seq_row["microarch"].values[0]:.1f}%)')
        plot_vtune_lines(ax2, par_rows, 'microarch')
        ax2.set_xlabel('Threads', fontsize=12)
        ax2.set_ylabel('Microarchitecture Usage (%)', fontsize=12)
        ax2.set_title('VTune: Microarch Usage vs Threads (maior = melhor)', fontsize=13)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(f'{OUTPUT_DIR}/vtune_sobreposicao_cpi_microarch.png', dpi=150)
        plt.close(fig)
        print(f"  -> {OUTPUT_DIR}/vtune_sobreposicao_cpi_microarch.png")

print()
print(f"Graficos salvos em: {OUTPUT_DIR}/")
print(f"Total: {len(os.listdir(OUTPUT_DIR))} arquivos")
