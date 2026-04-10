"""
Özellik Seçme Pipeline'ı
========================
Kötücül JavaScript Tespiti — Makine Öğrenmesi Projesi

Adımlar:
  1. VT + Korelasyon filtresi → 114'ten 98'e
  2. 4 bağımsız yöntem (MI, Chi2, ANOVA, RFE) → her biri top-40
  3. Downstream RF ile karşılaştır → en yüksek F1 kazanır
  4. Kazanan yöntemin 40 feature'ı CSV olarak kaydedilir

Çıktılar (fs_output/ klasörüne):
  features_selected.csv
  01_azalma.png          — 114 → 107 → 98 → 40
  02_karsilastirma.png   — 4 yöntemin F1/AUC karşılaştırması
  03_mi_rf_top50.png     — MI ve RF top-50 + kesişim grafiği
  04_grup_dagilimi.png   — seçilen 40'ın grup dağılımı
  yontem_sonuclari.csv
"""

import os, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import (VarianceThreshold, mutual_info_classif,
                                        SelectKBest, chi2, f_classif, RFE)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score
warnings.filterwarnings('ignore')

# ── Ayarlar ───────────────────────────────────────────────────────────────
INPUT_CSV    = 'features.csv'
OUTPUT_DIR   = 'fs_output'
TOP_N        = 40
RANDOM_STATE = 42
EVAL_STATE   = 99   # downstream RF — selection'dan bağımsız
os.makedirs(OUTPUT_DIR, exist_ok=True)

GROUP_COLORS = {
    'Yapısal':    '#FF9800',
    'Karakter':   '#00BCD4',
    'Gizleme':    '#4CAF50',
    'API':        '#2196F3',
    'Kod Yapısı': '#9C27B0',
    'Ağ/URL':     '#E05C3A',
    'Diğer':      '#9E9E9E',
}

GROUPS = {
    'Yapısal':    ['file_length','num_lines','avg_line_length','max_line_length',
                   'count_very_long_lines'],
    'Karakter':   ['ratio_alpha','ratio_digit','ratio_space','ratio_special','ratio_uppercase',
                   'count_semicolon','count_paren','count_bracket','count_square','count_plus',
                   'count_pipe','count_percent','count_single_quote','count_double_quote',
                   'count_string_literals','count_block_comments','unique_chars'],
    'Gizleme':    ['entropy_full','entropy_strings','hex_encoded_chars','count_str_concat',
                   'count_bracket_access','unique_short_varnames','obfusc_score'],
    'API':        ['api_eval','api_innerHTML','api_document_write','api_src_assign',
                   'api_atob','api_fromCharCode','api_density','dangerous_api_total',
                   'dangerous_api_diversity'],
    'Kod Yapısı': ['count_function_decl','count_function_expr','count_for','count_while',
                   'count_if','count_try_catch','count_var','count_this',
                   'max_nesting_depth','count_ternary','count_regex'],
    'Ağ/URL':     ['count_urls','count_suspicious_tld'],
}
feat_to_group = {f: g for g, feats in GROUPS.items() for f in feats}

# ══════════════════════════════════════════════════════════════════════════
# 1. VERİ YÜKLEME & SPLIT
# ══════════════════════════════════════════════════════════════════════════
print('='*60)
print('  ÖZELLİK SEÇME YÖNTEMLERİ KARŞILAŞTIRMASI')
print('='*60)

df    = pd.read_csv(INPUT_CSV)
X_raw = df.drop(columns=['filename','label']).reset_index(drop=True)
y     = df['label'].reset_index(drop=True)
n_start = X_raw.shape[1]

print(f'\n📂 Yüklendi: {df.shape}')
print(f'   Benign: {(y==0).sum():,}  |  Malicious: {(y==1).sum():,}')

X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
train_idx = X_train.index.tolist()
test_idx  = X_test.index.tolist()
print(f'   Train: {len(X_train):,}  |  Test: {len(X_test):,}')

# ══════════════════════════════════════════════════════════════════════════
# 2. ÖN FİLTRELEME  (sadece train'e fit)
# ══════════════════════════════════════════════════════════════════════════
# Variance Threshold
vt = VarianceThreshold(threshold=0.01)
vt.fit(X_train)
kept_vt    = X_train.columns[vt.get_support()].tolist()
n_after_vt = len(kept_vt)
Xtr_vt     = pd.DataFrame(vt.transform(X_train), columns=kept_vt)
Xte_vt     = pd.DataFrame(vt.transform(X_test),  columns=kept_vt)

# Korelasyon filtresi
corr    = Xtr_vt.corr().abs()
upper   = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
to_drop = [c for c in upper.columns if any(upper[c] > 0.95)]
clean   = [f for f in kept_vt if f not in to_drop]
n_after_corr = len(clean)

X_tr = Xtr_vt[clean].reset_index(drop=True)
X_te = Xte_vt[clean].reset_index(drop=True)

print(f'\n📉 Variance Threshold  : {n_start} → {n_after_vt}  (elinen: {n_start - n_after_vt})')
print(f'📉 Korelasyon filtresi : {n_after_vt} → {n_after_corr}  (elinen: {n_after_vt - n_after_corr})')
print(f'   → {n_after_corr} feature 4 yöntem için ortak başlangıç\n')

# ══════════════════════════════════════════════════════════════════════════
# 3. 4 BAĞIMSIZ YÖNTEM  (sadece train üzerinde)
# ══════════════════════════════════════════════════════════════════════════

# MI
print('⏳ Mutual Information hesaplanıyor...')
mi_scores = mutual_info_classif(X_tr, y_train, random_state=RANDOM_STATE)
mi_ser    = pd.Series(mi_scores, index=X_tr.columns).sort_values(ascending=False)
mi_top40  = mi_ser.head(TOP_N).index.tolist()

# Chi-Square (MinMaxScaler — negatif değer kabul etmez)
print('⏳ Chi-Square hesaplanıyor...')
mm         = MinMaxScaler()
Xtr_mm     = mm.fit_transform(X_tr)
chi_sel    = SelectKBest(chi2, k=TOP_N)
chi_sel.fit(Xtr_mm, y_train)
chi_scores = pd.Series(chi_sel.scores_, index=X_tr.columns).sort_values(ascending=False)
chi_top40  = X_tr.columns[chi_sel.get_support()].tolist()

# ANOVA
print('⏳ ANOVA F-score hesaplanıyor...')
anova_sel    = SelectKBest(f_classif, k=TOP_N)
anova_sel.fit(X_tr, y_train)
anova_scores = pd.Series(anova_sel.scores_, index=X_tr.columns).sort_values(ascending=False)
anova_top40  = X_tr.columns[anova_sel.get_support()].tolist()

# RFE
print('⏳ RFE çalışıyor (bu biraz uzun sürer)...')
rf_rfe  = RandomForestClassifier(n_estimators=50, class_weight='balanced',
                                  random_state=RANDOM_STATE, n_jobs=-1)
rfe     = RFE(rf_rfe, n_features_to_select=TOP_N, step=5)
rfe.fit(X_tr, y_train)
rfe_top40 = X_tr.columns[rfe.support_].tolist()
rfe_scores = pd.Series(rfe.ranking_, index=X_tr.columns)

# RF Importance — grafik 3 için (MI ile karşılaştırma)
print('⏳ RF Importance hesaplanıyor...')
rf_imp = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                random_state=RANDOM_STATE, n_jobs=-1)
rf_imp.fit(X_tr, y_train)
rf_ser   = pd.Series(rf_imp.feature_importances_, index=X_tr.columns).sort_values(ascending=False)
rf_top40 = rf_ser.head(TOP_N).index.tolist()
# NOT: rf_ser ve rf_top40 sadece Grafik 3 (MI∩RF top-50) için kullanılıyor.
# Downstream karşılaştırmasına dahil edilmiyor — RFE zaten RF tabanlı eleme yaptığı için
# RF Importance'ı ayrıca eklemek döngüselliği artırır.

METHODS = {
    'MI':    mi_top40,
    'Chi2':  chi_top40,
    'ANOVA': anova_top40,
    'RFE':   rfe_top40,
}

# ══════════════════════════════════════════════════════════════════════════
# 4. DOWNSTREAM DEĞERLENDİRME  (RF random_state=99)
# ══════════════════════════════════════════════════════════════════════════
def evaluate(feats, label):
    clf = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                 random_state=EVAL_STATE, n_jobs=-1)
    clf.fit(X_tr[feats], y_train)
    pred  = clf.predict(X_te[feats])
    proba = clf.predict_proba(X_te[feats])[:, 1]
    return {
        'Yöntem':    label,
        'N':         len(feats),
        'F1':        round(f1_score(y_test, pred), 5),
        'AUC':       round(roc_auc_score(y_test, proba), 5),
        'Accuracy':  round(accuracy_score(y_test, pred), 5),
        'Precision': round(precision_score(y_test, pred), 5),
        'Recall':    round(recall_score(y_test, pred), 5),
    }

print(f'\n{"─"*55}')
print(f'⏳ Downstream değerlendirme  [RF random_state={EVAL_STATE}]')
print(f'{"─"*55}')

results = [evaluate(X_tr.columns.tolist(), f'Baseline ({n_after_corr})')]
print(f'  {f"Baseline ({n_after_corr})":<18} | F1={results[0]["F1"]:.5f} | AUC={results[0]["AUC"]:.5f}')

for name, feats in METHODS.items():
    r = evaluate(feats, name)
    results.append(r)
    print(f'  {name:<18} | F1={r["F1"]:.5f} | AUC={r["AUC"]:.5f}')

res_df = pd.DataFrame(results).sort_values(['F1','AUC'], ascending=False).reset_index(drop=True)
res_df.to_csv(os.path.join(OUTPUT_DIR, 'yontem_sonuclari.csv'), index=False)

# En iyi yöntem (Baseline hariç) — F1 eşitse AUC'a bak
best_row  = res_df[~res_df['Yöntem'].str.startswith('Baseline')].iloc[0]
best_name = best_row['Yöntem']
final_feats = METHODS[best_name]

print(f'\n🏆 Kazanan: {best_name}  (F1={best_row["F1"]:.5f}  AUC={best_row["AUC"]:.5f})')

# ══════════════════════════════════════════════════════════════════════════
# 5. CSV KAYDET
# ══════════════════════════════════════════════════════════════════════════
df_tr             = X_tr[final_feats].copy()
df_tr['label']    = y_train.values
df_tr['filename'] = df['filename'].iloc[train_idx].values
df_tr['split']    = 'train'

df_te             = X_te[final_feats].copy()
df_te['label']    = y_test.values
df_te['filename'] = df['filename'].iloc[test_idx].values
df_te['split']    = 'test'

df_out = pd.concat([df_tr, df_te], ignore_index=True)
out_csv = os.path.join(OUTPUT_DIR, 'features_selected.csv')
df_out.to_csv(out_csv, index=False)
print(f'💾 {out_csv} → {df_out.shape}')

# ══════════════════════════════════════════════════════════════════════════
# 6. GRAFİKLER
# ══════════════════════════════════════════════════════════════════════════

# ── Grafik 1: 114 → 107 → 98 → 40 ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
stages  = [
    ('Başlangıç\n(ham)',         n_start,      '#78909C'),
    ('Variance\nThreshold',      n_after_vt,   '#546E7A'),
    ('Korelasyon\nFiltresi',     n_after_corr, '#37474F'),
    (f'★ Final\n({best_name})',  TOP_N,        '#FFD700'),
]
x_pos = range(len(stages))
for i, (label, val, color) in enumerate(stages):
    ax.bar(i, val, color=color, edgecolor='white', width=0.55, zorder=3)
    ax.text(i, val + 1.5, str(val), ha='center', fontsize=14, fontweight='bold')

# oklar
for i in range(len(stages) - 1):
    ax.annotate('', xy=(i + 0.62, stages[i+1][1] * 0.55),
                xytext=(i + 0.38, stages[i][1] * 0.55),
                arrowprops=dict(arrowstyle='->', color='#90A4AE', lw=2))

ax.set_xticks(x_pos)
ax.set_xticklabels([s[0] for s in stages], fontsize=11)
ax.set_ylabel('Feature Sayısı', fontsize=11)
ax.set_ylim(0, n_start * 1.2)
ax.set_title(f'Feature Sayısının Adım Adım Azalması  ({n_start} → {TOP_N})\n'
             f'Kazanan yöntem: {best_name}', fontsize=13, fontweight='bold')
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '01_azalma.png'), dpi=150, bbox_inches='tight')
plt.close()
print('\n✅ 01_azalma.png')

# ── Grafik 2: Yöntem karşılaştırma ────────────────────────────────────────
METHOD_COLORS = {
    'Baseline (98)': '#BDBDBD',
    'MI':   '#2196F3',
    'Chi2': '#FF9800',
    'ANOVA':'#E91E63',
    'RFE':  '#4CAF50',
}

fig = plt.figure(figsize=(14, 9))
gs  = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], hspace=0.5, wspace=0.4)

# Üst: sayısal tablo
ax_tbl = fig.add_subplot(gs[0, :])
ax_tbl.axis('off')
cols = ['Yöntem', 'N Feature', 'F1', 'AUC', 'Accuracy', 'Precision', 'Recall']
rows = []
for _, row in res_df.iterrows():
    mark = ' ★' if row['Yöntem'] == best_name else ''
    rows.append([row['Yöntem'] + mark, int(row['N']),
                 f"{row['F1']:.5f}", f"{row['AUC']:.5f}",
                 f"{row['Accuracy']:.5f}", f"{row['Precision']:.5f}", f"{row['Recall']:.5f}"])
tbl = ax_tbl.table(cellText=rows, colLabels=cols, loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.9)
for j in range(len(cols)):
    tbl[0, j].set_facecolor('#37474F')
    tbl[0, j].set_text_props(color='white', fontweight='bold')
for i, row in enumerate(rows):
    for j in range(len(cols)):
        if '★' in str(row[0]):
            tbl[i+1, j].set_facecolor('#FFF9C4')
            tbl[i+1, j].set_text_props(fontweight='bold')
        else:
            tbl[i+1, j].set_facecolor('#F5F5F5' if i % 2 == 0 else 'white')
ax_tbl.set_title(f'Downstream RF Değerlendirmesi  (random_state={EVAL_STATE})  |  ★ = Kazanan yöntem',
                 fontsize=11, fontweight='bold', pad=10)

# Sol alt: 1-AUC
ax_auc = fig.add_subplot(gs[1, 0])
order  = res_df['Yöntem'].tolist()
e_auc  = [1 - res_df.loc[res_df['Yöntem']==n, 'AUC'].values[0] for n in order]
bcols  = [METHOD_COLORS.get(n, '#BDBDBD') for n in order]
bars   = ax_auc.barh(order, e_auc, color=bcols, edgecolor='white', height=0.6)
for bar, val in zip(bars, e_auc):
    ax_auc.text(bar.get_width()*1.01, bar.get_y()+bar.get_height()/2,
                f'{val:.5f}', va='center', fontsize=9)
ax_auc.set_title('1 − AUC  (hata oranı, küçük = iyi)', fontsize=10, fontweight='bold')
ax_auc.set_xlabel('1 − AUC')
ax_auc.invert_yaxis()
ax_auc.spines[['top', 'right']].set_visible(False)

# Sağ alt: F1
ax_f1 = fig.add_subplot(gs[1, 1])
f1v   = [res_df.loc[res_df['Yöntem']==n, 'F1'].values[0] for n in order]
mn, mx = min(f1v), max(f1v)
bars2  = ax_f1.barh(order, f1v, color=bcols, edgecolor='white', height=0.6)
ax_f1.set_xlim(mn - (mx-mn)*5, mx + (mx-mn)*0.5)
for bar, val in zip(bars2, f1v):
    ax_f1.text(bar.get_width()+(mx-mn)*0.1, bar.get_y()+bar.get_height()/2,
               f'{val:.5f}', va='center', fontsize=9)
ax_f1.set_title('F1 Skoru  (büyük = iyi)', fontsize=10, fontweight='bold')
ax_f1.set_xlabel('F1')
ax_f1.invert_yaxis()
ax_f1.spines[['top', 'right']].set_visible(False)

plt.savefig(os.path.join(OUTPUT_DIR, '02_karsilastirma.png'), dpi=150, bbox_inches='tight')
plt.close()
print('✅ 02_karsilastirma.png')

# ── Grafik 3: MI top-50 / RF top-50 / Kesişim ─────────────────────────────
mi50_set = set(mi_ser.head(50).index)
rf50_set = set(rf_ser.head(50).index)
common   = mi50_set & rf50_set

fig, axes = plt.subplots(1, 3, figsize=(20, 13))

# Sol: MI top-50
mi_top50  = mi_ser.head(50)
mi_colors = ['#9C27B0' if f in common else '#2196F3' for f in mi_top50.index]
axes[0].barh(range(len(mi_top50)), mi_top50.values, color=mi_colors, edgecolor='white', height=0.8)
axes[0].set_yticks(range(len(mi_top50)))
axes[0].set_yticklabels(mi_top50.index, fontsize=8)
for i, (f, val) in enumerate(mi_top50.items()):
    axes[0].text(val * 1.005, i, f'{val:.3f}', va='center', fontsize=7)
axes[0].invert_yaxis()
axes[0].set_xlabel('MI Skoru')
axes[0].set_title(f'MI — Top 50\n(mor = RF top-50 ile ortak  [{len(common)}])',
                  fontsize=11, fontweight='bold')
axes[0].spines[['top', 'right']].set_visible(False)

# Orta: RF top-50
rf_top50  = rf_ser.head(50)
rf_colors = ['#9C27B0' if f in common else '#4CAF50' for f in rf_top50.index]
axes[1].barh(range(len(rf_top50)), rf_top50.values, color=rf_colors, edgecolor='white', height=0.8)
axes[1].set_yticks(range(len(rf_top50)))
axes[1].set_yticklabels(rf_top50.index, fontsize=8)
for i, (f, val) in enumerate(rf_top50.items()):
    axes[1].text(val * 1.005, i, f'{val:.4f}', va='center', fontsize=7)
axes[1].invert_yaxis()
axes[1].set_xlabel('RF Importance (Gini)')
axes[1].set_title(f'RF — Top 50\n(mor = MI top-50 ile ortak  [{len(common)}])',
                  fontsize=11, fontweight='bold')
axes[1].spines[['top', 'right']].set_visible(False)

# Sağ: Kesişim
common_list = [f for f in mi_ser.index if f in common]
mi_n = (mi_ser[common_list] / mi_ser.max()).values
rf_n = (rf_ser[common_list] / rf_ser.max()).values
yp   = range(len(common_list))
axes[2].barh([y + 0.2 for y in yp], mi_n, height=0.38,
             color='#2196F3', alpha=0.85, label='MI skoru (normalize)')
axes[2].barh([y - 0.2 for y in yp], rf_n, height=0.38,
             color='#4CAF50', alpha=0.75, label='RF importance (normalize)')
axes[2].set_yticks(yp)
axes[2].set_yticklabels(common_list, fontsize=8)
axes[2].invert_yaxis()
axes[2].set_xlabel('Normalize Skor')
axes[2].set_title(f'MI ∩ RF Kesişimi — {len(common_list)} Feature\n(her ikisinin top-50\'sinde ortak)',
                  fontsize=11, fontweight='bold')
axes[2].legend(fontsize=9)
axes[2].spines[['top', 'right']].set_visible(False)

fig.legend(handles=[
    mpatches.Patch(color='#9C27B0', label=f'Ortak (MI∩RF top-50)  [{len(common)}]'),
    mpatches.Patch(color='#2196F3', label=f'Sadece MI  [{len(mi50_set - rf50_set)}]'),
    mpatches.Patch(color='#4CAF50', label=f'Sadece RF  [{len(rf50_set - mi50_set)}]'),
], loc='lower center', ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.01))
fig.suptitle('MI Top-50  /  RF Top-50  /  Ortak Kesişim Analizi',
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig(os.path.join(OUTPUT_DIR, '03_mi_rf_top50.png'), dpi=150, bbox_inches='tight')
plt.close()
print('✅ 03_mi_rf_top50.png')

# ── Grafik 4: Grup dağılımı (yüklenen görseldeki format) ─────────────────
group_counts = {}
for f in final_feats:
    g = feat_to_group.get(f, 'Diğer')
    group_counts[g] = group_counts.get(g, 0) + 1

df_grp = (pd.DataFrame(list(group_counts.items()), columns=['Grup', 'N'])
            .sort_values('N', ascending=True)
            .reset_index(drop=True))

group_descs = {
    'Yapısal':    'Dosya boyutu, satır sayısı',
    'Karakter':   'Karakter oranları ve sayımlar',
    'Gizleme':    'Obfuscation, encoding, entropy',
    'API':        'Tehlikeli JS API kullanımları',
    'Kod Yapısı': 'Fonksiyon, döngü, kontrol akışı',
    'Ağ/URL':     'Ağ, URL, IP özellikleri',
    'Diğer':      'Sınıflandırılmamış',
}

bar_colors = [GROUP_COLORS.get(g, '#9E9E9E') for g in df_grp['Grup']]

fig, ax = plt.subplots(figsize=(11, max(5, len(df_grp) * 0.85)))
bars = ax.barh(df_grp['Grup'], df_grp['N'],
               color=bar_colors, alpha=0.88, edgecolor='white', height=0.6)
for bar, (_, row) in zip(bars, df_grp.iterrows()):
    ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
            f'{int(row["N"])}', va='center', fontsize=12, fontweight='bold')

ax.set_xlim(0, df_grp['N'].max() * 1.25)
ax.set_xlabel('Feature Sayısı', fontsize=12)
ax.set_title(f'Seçilen {TOP_N} Feature — Tematik Grup Dağılımı\n(Kazanan yöntem: {best_name})',
             fontsize=13, fontweight='bold')
ax.axvline(x=df_grp['N'].max() * 0.9, color='gray', linestyle='--', alpha=0.3)
ax.grid(axis='x', linestyle='--', alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)

# Sağ tarafta açıklamalar
ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim())
ax2.set_yticks(range(len(df_grp)))
ax2.set_yticklabels([group_descs.get(g, '') for g in df_grp['Grup']], fontsize=9)
ax2.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
ax2.tick_params(right=False)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '04_grup_dagilimi.png'), dpi=150, bbox_inches='tight')
plt.close()
print('✅ 04_grup_dagilimi.png')

# ══════════════════════════════════════════════════════════════════════════
# 7. ÖZET
# ══════════════════════════════════════════════════════════════════════════
print(f"""
{'='*60}
  FEATURE SELECTION TAMAMLANDI ✅
{'='*60}
  Başlangıç       : {n_start} feature
  VT + Korelasyon : {n_after_corr} feature
  Kazanan yöntem  : {best_name}
  F1              : {best_row['F1']:.5f}
  AUC             : {best_row['AUC']:.5f}
  Final           : {len(final_feats)} feature

  Grup dağılımı:""")
for _, row in df_grp.sort_values('N', ascending=False).iterrows():
    feats_in = [f for f in final_feats if feat_to_group.get(f, 'Diğer') == row['Grup']]
    print(f'    {row["Grup"]:<15} {int(row["N"]):2d}  →  {", ".join(feats_in[:3])}{"..." if row["N"]>3 else ""}')

print(f"""
  Seçilen {len(final_feats)} feature ({best_name}):""")
for i, f in enumerate(final_feats, 1):
    g = feat_to_group.get(f, 'Diğer')
    print(f'  {i:2d}. {f:<35} [{g}]')

print(f"""
  Çıktılar → {OUTPUT_DIR}/
  ✅ features_selected.csv
  ✅ yontem_sonuclari.csv
  ✅ 01_azalma.png
  ✅ 02_karsilastirma.png
  ✅ 03_mi_rf_top50.png
  ✅ 04_grup_dagilimi.png
{'='*60}
""")
