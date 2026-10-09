"""Build report summaries and the five-specimen benchmark from recorded metrics."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports'
main = json.loads((ROOT / 'results/expanded/five_specimens/metrics.json').read_text())
continuity = json.loads((ROOT / 'results/expanded/five_specimens/continuity_sensitivity.json').read_text())
external = json.loads((ROOT / 'results/expanded/v1_external/metrics.json').read_text())
audit = json.loads((REPORTS / 'expanded_data_audit.json').read_text())
metadata = {d['specimen_id']: d for d in audit['specimens']}
folds = {f['test_extractions'][0]: f for f in main['folds']}
order = ['Probe1_22', 'Probe2_7', 'Exp3', 'Exp4', 'Exp5']

def fmt(value, decimals=2):
    return f'{value:.{decimals}f}'.replace('.', ',')

def values(sid, kind, metric='rmse_normalized'):
    f = folds[sid]
    if kind == 'phase_affine':
        return np.array([f['baselines'][kind][sid]['compression_window'][metric]])
    return np.array([s['tests'][sid]['compression_window'][metric] for s in f['models'][kind]])

def mean_sd(v):
    return float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0

aggregate_rows = []
for kind, label in [('gnn', 'GNN'), ('mlp', 'MLP sem mensagens'), ('phase_affine', 'Modelo afim')]:
    a = main['aggregate'][kind]
    norm = fmt(100 * a['macro_rmse_normalized_mean'])
    px = fmt(a['macro_rmse_px_mean'])
    if kind != 'phase_affine':
        norm += ' ± ' + fmt(100 * a['macro_rmse_normalized_sd_over_seeds'])
        px += ' ± ' + fmt(a['macro_rmse_px_sd_over_seeds'])
    aggregate_rows.append([label, norm + '%', px + ' px'])

fold_rows = []
for sid in order:
    mn, sd = mean_sd(values(sid, 'gnn') * 100)
    fold_rows.append([sid, fmt(mn) + ' ± ' + fmt(sd), fmt(values(sid, 'phase_affine')[0] * 100), fmt(metadata[sid]['compression_coverage'] * 100, 1) + '%'])

gnn = main['aggregate']['gnn']['macro_rmse_normalized_mean']
affine = main['aggregate']['phase_affine']['macro_rmse_normalized_mean']
reduction = (1 - gnn / affine) * 100
new_three = np.array([values(sid, 'gnn') for sid in ['Exp3', 'Exp4', 'Exp5']])
new_three_mean = float(new_three.mean())
new_three_sd = float(new_three.mean(axis=0).std(ddof=1))
old_three = external['aggregate']['test']['macro_rmse_normalized']
external_reduction = (1 - new_three_mean / old_three) * 100
retained = sum(s['retained_compression_node_frames'] for s in continuity['specimens'])
original = sum(s['source_observed_compression_node_frames'] for s in continuity['specimens'])
cs_gnn = continuity['aggregate']['gnn']['macro_rmse_normalized_mean']
cs_affine = continuity['aggregate']['phase_affine']['macro_rmse_normalized_mean']
cs_reduction = (1 - cs_gnn / cs_affine) * 100
summary_path = REPORTS / 'expansion_summary.json'
previous = json.loads(summary_path.read_text()) if summary_path.exists() else {}
summary = {
    # Preserve finalized augmentation results and any other report extensions.
    **previous,
    'headline': f'<b>A GNN obteve {fmt(100 * gnn)}% da altura inicial de RMSE, contra {fmt(100 * affine)}% do modelo afim: redução relativa de {fmt(reduction)}%.</b> A avaliação usa cinco espécimes reservados, um por fold, com três seeds fixas. A cobertura desigual do tracking limita a interpretação aos nós observados.',
    'specimens': [{k: metadata[sid][k] for k in ['specimen_id', 'n_nodes', 'n_undirected_edges', 'compression_frames', 'compression_observed_node_frames', 'compression_coverage']} for sid in order],
    'aggregate_rows': aggregate_rows,
    'fold_rows': fold_rows,
    'result_interpretation': f'A redução relativa de {fmt(reduction)}% face à base afim refere-se à média do RMSE normalizado com o mesmo peso por espécime. A GNN apresenta menor erro médio nos cinco folds; a MLP permanece próxima da base afim. Isto apoia a utilidade das mensagens entre nós nesta amostra, mas não estabelece generalização a outras campanhas ou materiais. Os valores ± medem apenas variação entre as três seeds.',
    'external_text': f'O checkpoint da primeira versão, treinado nos dois ensaios originais e congelado, obteve {fmt(100 * old_three, 4)}% de RMSE normalizado nos três ensaios adicionais. Nos mesmos três ensaios, cada um reservado no seu fold do novo protocolo, a GNN obteve {fmt(100 * new_three_mean, 4)}% ± {fmt(100 * new_three_sd, 4)}% entre seeds: erro médio {fmt(external_reduction)}% inferior. As máscaras de avaliação são as mesmas; os conjuntos de ajuste e a escolha de época mudaram. Esta comparação é descritiva e não uma ablação controlada do tamanho do dataset.',
    'verification_text': 'Verificações concluídas: os três novos pares grafo/tracking e as três tabelas de anotações, juntamente com 431 imagens, foram conferidos contra os hashes da origem; os cinco datasets passaram os controlos de máscaras, índices e simetria das arestas. Uma extração independente de Probe1_8 reproduziu o hash agregado dos píxeis dos 2 048 frames. O teste de inferência nas quatro geometrias automáticas gerou 1 545 previsões nó-fase finitas usando apenas os cinco campos de geometria inicial; um controlo de acesso impede a leitura dos alvos. Este último teste verifica a execução da inferência, não a sua precisão.',
    'automatic_caption': previous.get('automatic_caption', 'PENDENTE: inserir avaliação final da curadoria automática.'),
    'automatic_decisions': previous.get('automatic_decisions', ['PENDENTE: decisões finais de admissão dos vídeos.']),
    'automatic_evaluation': previous.get('automatic_evaluation', ['PENDENTE: resultados finais da avaliação adicional dos vídeos.']),
    'numeric_audit': {
        'gnn_relative_reduction_vs_affine_percent': reduction,
        'v1_external_three_macro_rmse_normalized': old_three,
        'expanded_three_macro_rmse_normalized': new_three_mean,
        'expanded_three_macro_rmse_normalized_sd_over_seeds': new_three_sd,
        'external_three_relative_reduction_percent': external_reduction,
        'continuity_retained_observations': retained,
        'continuity_original_observations': original,
        'continuity_relative_reduction_vs_affine_percent': cs_reduction,
    },
}
if 'verification_text' in previous:
    # Final verification can include later checks outside this five-fold summary.
    summary['verification_text'] = previous['verification_text']
retained_text = f'{retained:,}'.replace(',', ' ')
original_text = f'{original:,}'.replace(',', ' ')
summary['continuity_text'] = f'Sem voltar a treinar nem escolher épocas, cada trajetória foi truncada permanentemente após a primeira lacuna com mais de dois frames intermédios sem observação (separação entre frames observados superior a três). Restaram {retained_text} de {original_text} observações. Nesse subconjunto, o RMSE normalizado é {fmt(100 * cs_gnn)}% para a GNN e {fmt(100 * cs_affine)}% para a base afim, uma redução relativa de {fmt(cs_reduction)}%. A conclusão agregada mantém-se, mas o filtro seleciona as trajetórias mais persistentes e não confirma por si a identidade física dos nós.'
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.labelcolor': '#17334c', 'text.color': '#17334c', 'xtick.color': '#17334c', 'ytick.color': '#566773'})
fig, (ax, cov) = plt.subplots(2, 1, figsize=(10, 5.0), sharex=True, gridspec_kw={'height_ratios': [3.4, 1], 'hspace': 0.14})
colors = {'gnn': '#246f99', 'mlp': '#bc994e', 'phase_affine': '#a3b1bb'}
x = np.arange(len(order))
width = 0.235
for i, (kind, label) in enumerate([('gnn', 'GNN'), ('mlp', 'MLP'), ('phase_affine', 'Afim')]):
    stats = np.array([mean_sd(values(sid, kind) * 100) for sid in order])
    ax.bar(x + (i - 1) * width, stats[:, 0], width, color=colors[kind], label=label, zorder=3, yerr=stats[:, 1], error_kw={'elinewidth': 1, 'capsize': 3, 'capthick': 1, 'ecolor': '#17334c'})
ax.set_ylabel('RMSE / altura inicial (%)')
ax.set_ylim(0, 28)
ax.set_yticks(np.arange(0, 26, 5))
ax.set_title('Teste com um espécime inteiro reservado', loc='left', fontsize=15, fontweight='bold', pad=16)
ax.legend(ncol=3, frameon=False, loc='upper right', fontsize=11)
ax.tick_params(axis='x', bottom=False)
coverage = np.array([metadata[sid]['compression_coverage'] * 100 for sid in order])
cov.bar(x, coverage, 0.76, color=['#cedde5' if c >= 90 else '#eee0ba' for c in coverage], zorder=3)
for xi, cv in zip(x, coverage):
    cov.text(xi, 5, f'{cv:.1f}%'.replace('.', ','), ha='center', va='bottom', fontsize=11, fontweight='bold')
cov.set_ylim(0, 111)
cov.set_yticks([0, 100])
cov.set_ylabel('Cobertura\ndo tracking (%)', fontsize=10)
cov.set_xticks(x, ['Probe1_22', 'Probe2_7', 'Exp3 / D12', 'Exp4 / D14', 'Exp5 / ID03'])
for a in [ax, cov]:
    a.grid(axis='y', color='#e1e6ea', linewidth=0.7, zorder=0)
    a.spines[['top', 'right']].set_visible(False)
    a.spines[['left', 'bottom']].set_color('#cdd6dc')
fig.subplots_adjust(left=.10, right=.99, top=.89, bottom=.11)
fig.savefig(REPORTS / 'figures/expanded_benchmark.png', dpi=220, facecolor='white')
fig.savefig(REPORTS / 'figures/expanded_benchmark.pdf', facecolor='white')
plt.close(fig)
print(json.dumps(summary['numeric_audit'], indent=2))
