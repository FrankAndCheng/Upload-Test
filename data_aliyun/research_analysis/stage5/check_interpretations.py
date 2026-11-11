"""复核论文中的区间、尾部诊断、改善比例和处理替换；不重训。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
checks = []

def compare(kind, key, metric, value, saved, tol=5e-7):
    error = abs(float(value) - float(saved))
    checks.append(dict(kind=kind, key=key, metric=metric, recomputed=float(value),
                       saved=float(saved), absolute_difference=error, passed=error <= tol))

def interval(delta, rng, n):
    samples = np.array([delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(n)])
    return delta.mean(), *np.quantile(samples, [.025, .975])

seq_path = ROOT / 'review/sequential_experiments'
pre_path = ROOT / 'review/preprocessing_ablation'
seq_ci = pd.read_csv(seq_path / 'paired_intervals.csv')
bands = pd.read_csv(seq_path / 'target_bands.csv')
relative = {}
roundtrip = []
evaluation = pd.read_csv(seq_path / 'evaluation.csv')
for dataset in ['development', 'B']:
    d = pd.read_csv(seq_path / f'{dataset}_predictions.csv')
    y = d.actual.to_numpy()
    delta = (y - d.nn_blend_06.to_numpy())**2 - (y - d.baseline_xgb.to_numpy())**2
    # 补充融合脚本对各集合重新初始化种子23。
    values = interval(delta, np.random.default_rng(23), 3000)
    saved = seq_ci[(seq_ci.dataset == dataset) & (seq_ci.name == 'nn_blend_06')].iloc[0]
    for metric, value in zip(['MSE_difference', 'low95', 'high95'], values):
        compare('final_interval', dataset, metric, value, saved[metric])
    for r in bands[bands.dataset == dataset].itertuples():
        lo, hi = r.band.split('-')
        select = (y >= float(lo)) & (y < float(hi))
        error = (y - d[r.name].to_numpy())**2
        values = dict(rows=select.sum(), MSE=error[select].mean(),
                      squared_error_share=error[select].sum()/error.sum(),
                      mean_prediction_minus_actual=(d[r.name].to_numpy()-y)[select].mean())
        for metric, value in values.items():
            compare('target_band', dataset + '/' + r.name + '/' + r.band, metric, value, getattr(r, metric))
    base = np.mean((y - d.baseline_xgb.to_numpy())**2)
    final = np.mean((y - d.nn_blend_06.to_numpy())**2)
    saved_metrics = evaluation[evaluation.dataset == dataset].set_index('name')
    saved_base = float(saved_metrics.loc['baseline_xgb', 'MSE'])
    saved_final = float(saved_metrics.loc['nn_blend_06', 'MSE'])
    # 正文的六位改善百分比由未舍入的保存指标计算；CSV预测往返存在已量化的小误差。
    relative[dataset] = 100 * (saved_base - saved_final) / saved_base
    compare('relative_reduction_percent', dataset, 'percent', relative[dataset],
            {'development': 2.313554, 'B': 1.472594}[dataset], tol=5e-7)
    csv_relative = 100 * (base - final) / base
    bound = 100 * (abs(final-saved_final)/base + abs(saved_final)*abs(saved_base-base)/(base*saved_base))
    roundtrip.append(dict(dataset=dataset, saved_metric_percent=relative[dataset],
                          CSV_prediction_percent=csv_relative, absolute_difference=abs(csv_relative-relative[dataset]),
                          propagated_roundtrip_bound=float(bound), passed=bool(abs(csv_relative-relative[dataset]) <= bound+1e-12)))

# 完整保留 render_results.py 的随机数消耗顺序，包括零差异的模型。
pre_ci = pd.read_csv(pre_path / 'holdout_paired_intervals.csv')
rec = pd.read_csv(pre_path / 'recommendations.csv')
rng = np.random.default_rng(23)
for dataset in ['development', 'B']:
    d = pd.read_csv(pre_path / f'{dataset}_predictions.csv')
    pairs = [(r.model, d[f'{r.model}__{r.recommended}'].to_numpy(),
              d[f'{r.model}__{r.reference}'].to_numpy()) for r in rec.itertuples()]
    pairs.append(('Fusion', d.Fusion__recommended.to_numpy(), d.Fusion__reference.to_numpy()))
    for model, candidate, reference in pairs:
        y = d.actual.to_numpy()
        delta = (y - candidate)**2 - (y - reference)**2
        values = interval(delta, rng, 2000)
        saved = pre_ci[(pre_ci.dataset == dataset) & (pre_ci.model == model)].iloc[0]
        for metric, value in zip(['MSE_delta', 'low95', 'high95'], values):
            compare('preprocessing_interval', dataset + '/' + model, metric, value, saved[metric])

cv = pd.read_csv(pre_path / 'cv_summary.csv')
fold = pd.read_csv(pre_path / 'fold_results.csv')
rule_checks = []
for r in rec.itertuples():
    sub = cv[cv.model == r.model]
    ref_score = float(sub[sub['mode'] == r.reference].CV_MSE.iloc[0])
    ref_fold = fold[(fold.model == r.model) & (fold['mode'] == r.reference)].sort_values(['seed','fold']).MSE.to_numpy()
    eligible = []
    for candidate in sub.itertuples():
        values = fold[(fold.model == r.model) & (fold['mode'] == candidate.mode)].sort_values(['seed','fold']).MSE.to_numpy()
        wins = int((values < ref_fold - 1e-12).sum())
        improvement = (ref_score - candidate.CV_MSE) / ref_score
        if candidate.mode == r.reference or (improvement >= .005 and wins >= 7):
            eligible.append((candidate.CV_MSE, candidate.mode))
    chosen = min(eligible)[1]
    rule_checks.append(dict(model=r.model, recomputed=chosen, saved=r.recommended, passed=chosen == r.recommended))

pd.DataFrame(checks).to_csv(OUT / 'interpretation_recalculation.csv', index=False, encoding='utf-8-sig')
(OUT / 'replacement_rule_checks.json').write_text(json.dumps(rule_checks, ensure_ascii=False, indent=2))
(OUT / 'relative_reduction_roundtrip.json').write_text(json.dumps(roundtrip, ensure_ascii=False, indent=2))
summary = dict(numeric_comparisons=len(checks), numeric_passed=sum(x['passed'] for x in checks),
               max_absolute_difference=max(x['absolute_difference'] for x in checks),
               replacement_rules=len(rule_checks), replacement_rules_passed=sum(x['passed'] for x in rule_checks),
               relative_reductions_percent=relative, retrained=False)
(OUT / 'interpretation_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print(json.dumps(summary, ensure_ascii=False, indent=2))
assert all(x['passed'] for x in checks), [x for x in checks if not x['passed']]
assert all(x['passed'] for x in rule_checks)
assert all(x['passed'] for x in roundtrip)
