"""按 paper.ipynb C002/C014/C017/C023/C034/C037 重排已有相关热图。

只重新计算原训练探索的同一矩阵用于图形排版，不训练预测模型。
原 PNG 保留。新图省略格内数值，完整40×40矩阵另存 CSV。
"""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/glucose-stage5-matplotlib')
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PAPER = ROOT / 'research_analysis/stage3/paper'
train = pd.read_csv(ROOT / 'data/d_train_20180102.csv')
a = pd.read_csv(ROOT / 'data/d_test_A_20180102.csv')
a['血糖'] = pd.read_csv(ROOT / 'data/d_answer_a_20180128.csv').iloc[:, 0].to_numpy()
full = pd.concat([train, a], ignore_index=True)
fit, _ = train_test_split(full, test_size=.3, random_state=23)
expected_ids = pd.read_csv(ROOT / 'review/sequential_experiments/split_ids.csv')
assert set(fit.id) == set(expected_ids.loc[expected_ids.split == 'train', 'id'])
x = fit.drop(columns=['id', '体检日期', '血糖']).copy()
x['性别'] = x['性别'].map({'女': 0., '男': 1.})
x['性别未知'] = x['性别'].isna().astype(float)
imputed = pd.DataFrame(SimpleImputer(strategy='median', keep_empty_features=True).fit_transform(x), columns=x.columns)
z = pd.DataFrame(StandardScaler().fit_transform(imputed), columns=x.columns)
corr = z.corr()
assert corr.shape == (40, 40) and len(fit) == 4649
assert '血糖' not in corr.columns and 'id' not in corr.columns and '体检日期' not in corr.columns
scaling_gap = float(np.nanmax(np.abs(corr.to_numpy() - imputed.corr().to_numpy())))
assert scaling_gap < 1e-12
corr.to_csv(PAPER / 'tables/training_correlation_matrix.csv', encoding='utf-8-sig', float_format='%.17g')
plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.rcParams['axes.unicode_minus'] = False
fig, ax = plt.subplots(figsize=(9.5, 9.6), dpi=300)
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap=sns.diverging_palette(250, 5, as_cmap=True),
            center=0, vmin=-1, vmax=1, square=True, annot=False, linewidths=.3,
            cbar_kws={'shrink': .72, 'ticks': [-1, -.5, 0, .5, 1], 'label': '相关系数'}, ax=ax)
ax.tick_params(axis='both', labelsize=8.4, length=0)
plt.setp(ax.get_xticklabels(), rotation=90, ha='center')
plt.setp(ax.get_yticklabels(), rotation=0)
ax.collections[0].colorbar.ax.tick_params(labelsize=9)
ax.collections[0].colorbar.ax.yaxis.label.set_size(10)
fig.subplots_adjust(left=.245, bottom=.25, right=.96, top=.98)
dest = PAPER / 'figures/training_correlation_reformatted.png'
fig.savefig(dest, dpi=300, bbox_inches='tight', pad_inches=.06)
plt.close(fig)
record = dict(notebook='paper.ipynb', cells_1based=[2,14,17,23,34,37],
              training_rows=4649, input_columns=40, target_included=False,
              change='Only layout: omit cell annotations, adjust label sizes, fix colorbar to [-1,1].',
              original_figure='research_analysis/stage1/figures/paper_C037_O02.png',
              rendered_figure=str(dest.relative_to(ROOT)), sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
              matrix_file='research_analysis/stage3/paper/tables/training_correlation_matrix.csv',
              standardized_vs_imputed_correlation_max_gap=scaling_gap,
              new_predictive_experiments=False)
(OUT / 'correlation_reformat.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
print(json.dumps(record, ensure_ascii=False, indent=2))
