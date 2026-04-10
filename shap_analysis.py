import os, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap

from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

INPUT_CSV    = 'features_selected.csv'
OUTPUT_DIR   = 'shap_outputs'
TOP_N        = 10
SAMPLE_SIZE  = 1000
KERNEL_BG    = 100
KERNEL_EVAL  = 200
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = {
    'Random Forest'      : '#E05C3A',
    'XGBoost'            : '#2196F3',
    'Decision Tree'      : '#FF9800',
    'KNN'                : '#9C27B0',
    'Logistic Regression': '#4CAF50',
    'SVM'                : '#00BCD4',
    'Naive Bayes'        : '#795548',
}

def plot_bar(shap_series, model_name, filename, color):
    top  = shap_series.head(TOP_N)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(top.index[::-1], top.values[::-1], color=color, alpha=0.85, height=0.6)
    for bar, val in zip(bars, top.values[::-1]):
        ax.text(val + top.values.max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', ha='left', fontsize=9)
    ax.set_xlabel('Ortalama |SHAP Değeri|', fontsize=11)
    ax.set_title(f'{model_name}\nSHAP Özellik Önemi (Top {TOP_N})',
                 fontsize=12, fontweight='bold', pad=12)
    ax.set_xlim(0, top.values.max() * 1.18)
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'   ✅ {path}')

def extract_class1(sv_raw, n_features):
    if isinstance(sv_raw, list):
        arr = np.array(sv_raw[1])
    else:
        arr = np.array(sv_raw)
    if arr.ndim == 3:
        arr = arr[:, :, 1]
    if arr.ndim == 2 and arr.shape[1] != n_features:
        arr = arr.T
    return arr

def mean_abs(sv, feat_names):
    vals = np.abs(sv).mean(axis=0)
    return pd.Series(vals, index=feat_names).sort_values(ascending=False)

print('\n' + '='*55)
print('  SHAP ANALİZİ BAŞLIYOR')
print('='*55)

print('\n📂 Veri yükleniyor...')
df = pd.read_csv(INPUT_CSV)

drop_cols = [c for c in ['label', 'filename', 'split'] if c in df.columns]
train_df  = df[df['split'] == 'train'].reset_index(drop=True)
test_df   = df[df['split'] == 'test'].reset_index(drop=True)

X_train = train_df.drop(columns=drop_cols)
y_train = train_df['label']
X_test  = test_df.drop(columns=drop_cols)
y_test  = test_df['label']
feats   = X_train.columns.tolist()

print(f'   Train: {len(X_train):,}  |  Test: {len(X_test):,}')
print(f'   Benign: {(y_train==0).sum():,}  |  Malicious: {(y_train==1).sum():,}')

scaler = StandardScaler()
Xtr    = pd.DataFrame(scaler.fit_transform(X_train), columns=feats)
Xte    = pd.DataFrame(scaler.transform(X_test),      columns=feats)

spw = (y_train == 0).sum() / (y_train == 1).sum()

print('\n🏋️  Modeller eğitiliyor...')
models = {
    'Random Forest'      : RandomForestClassifier(
                               n_estimators=200, class_weight='balanced',
                               random_state=RANDOM_STATE, n_jobs=-1),
    'XGBoost'            : XGBClassifier(
                               n_estimators=200, scale_pos_weight=spw,
                               eval_metric='logloss', verbosity=0,
                               random_state=RANDOM_STATE, n_jobs=-1),
    'Decision Tree'      : DecisionTreeClassifier(
                               class_weight='balanced', max_depth=20,
                               random_state=RANDOM_STATE),
    'KNN'                : KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    'Logistic Regression': LogisticRegression(
                               max_iter=1000, class_weight='balanced',
                               random_state=RANDOM_STATE, n_jobs=-1),
    'SVM'                : SVC(kernel='rbf', class_weight='balanced',
                               probability=True, random_state=RANDOM_STATE),
    'Naive Bayes'        : GaussianNB(),
}

for name, model in models.items():
    print(f'   ⏳ {name}...', end=' ', flush=True)
    model.fit(Xtr, y_train)
    print(f'Accuracy: {model.score(Xte, y_test):.4f}')

rng     = np.random.default_rng(RANDOM_STATE)
idx     = rng.choice(len(Xte), size=min(SAMPLE_SIZE, len(Xte)), replace=False)
X_shap  = Xte.iloc[idx]
bg_data = shap.sample(Xtr, KERNEL_BG, random_state=RANDOM_STATE)
X_eval  = X_shap.iloc[:KERNEL_EVAL]

rankings = {}
file_map = {
    'Random Forest'      : '01_rf_shap_bar.png',
    'XGBoost'            : '02_xgb_shap_bar.png',
    'Decision Tree'      : '03_dt_shap_bar.png',
    'KNN'                : '04_knn_shap_bar.png',
    'Logistic Regression': '05_lr_shap_bar.png',
    'SVM'                : '06_svm_shap_bar.png',
    'Naive Bayes'        : '07_nb_shap_bar.png',
}

for mname in ['Random Forest', 'XGBoost', 'Decision Tree']:
    print(f'\n🌲 {mname} SHAP (TreeExplainer)...')
    exp = shap.TreeExplainer(models[mname])
    sv  = extract_class1(exp.shap_values(X_shap), len(feats))
    ser = mean_abs(sv, feats)
    rankings[mname] = ser
    plot_bar(ser, mname, file_map[mname], COLORS[mname])

for mname in ['KNN', 'Logistic Regression', 'SVM', 'Naive Bayes']:
    print(f'\n🔍 {mname} SHAP (KernelExplainer)...')
    exp    = shap.KernelExplainer(models[mname].predict_proba, bg_data)
    sv_raw = exp.shap_values(X_eval, nsamples=100)
    sv     = extract_class1(sv_raw, len(feats))
    ser    = mean_abs(sv, feats)
    rankings[mname] = ser
    plot_bar(ser, mname, file_map[mname], COLORS[mname])

print('\n📊 Karşılaştırma grafiği hazırlanıyor (global normalize)...')
shap_df    = pd.DataFrame({m: rankings[m] for m in rankings})
global_max = shap_df.values.max()
ref_feats  = rankings['Random Forest'].head(TOP_N).index.tolist()
model_list = list(rankings.keys())
n_models   = len(model_list)
x          = np.arange(TOP_N)
bw         = 0.10

fig, ax = plt.subplots(figsize=(14, 6))
for i, mname in enumerate(model_list):
    ser    = rankings[mname]
    vals   = np.array([ser.get(f, 0) for f in ref_feats])
    norm   = vals / global_max
    offset = (i - n_models / 2 + 0.5) * bw
    ax.bar(x + offset, norm, bw * 0.92, color=COLORS[mname], alpha=0.85, label=mname)
ax.set_xticks(x)
ax.set_xticklabels(ref_feats, rotation=35, ha='right', fontsize=9)
ax.set_ylabel('Global Normalize SHAP Önemi', fontsize=11)
ax.set_title('7 Model — SHAP Özellik Önemi Karşılaştırması (Global Normalize)',
             fontsize=13, fontweight='bold', pad=14)
ax.legend(loc='upper right', fontsize=9, framealpha=0.9, ncol=2)
ax.grid(axis='y', linestyle='--', alpha=0.35)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
cmp_path = os.path.join(OUTPUT_DIR, '08_tum_modeller_karsilastirma.png')
plt.savefig(cmp_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'   ✅ {cmp_path}')

print('\n📊 Konsensüs analizi hazırlanıyor...')
vote_counts = {}
for mname, ser in rankings.items():
    for feat in ser.head(TOP_N).index:
        vote_counts[feat] = vote_counts.get(feat, 0) + 1
vote_ser       = pd.Series(vote_counts).sort_values(ascending=False)
n_models_total = len(rankings)
palette        = plt.cm.RdYlGn(np.linspace(0.25, 0.85, n_models_total))
bar_colors     = [palette[v - 1] for v in vote_ser.values]

fig, ax = plt.subplots(figsize=(10, max(6, len(vote_ser) * 0.45)))
bars = ax.barh(vote_ser.index[::-1], vote_ser.values[::-1],
               color=bar_colors[::-1], edgecolor='white', height=0.6)
for bar, val in zip(bars, vote_ser.values[::-1]):
    ax.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
            f'{val}/{n_models_total} model', va='center', fontsize=9)
ax.set_xlabel('Kaç model top-10\'da seçti?', fontsize=11)
ax.set_xlim(0, n_models_total + 1.5)
ax.set_xticks(range(0, n_models_total + 1))
ax.set_title('Konsensüs Analizi — Ortak Seçilen Özellikler',
             fontsize=13, fontweight='bold', pad=12)
ax.axvline(x=n_models_total,       color='green',  linestyle='--', alpha=0.5)
ax.axvline(x=n_models_total * 0.7, color='orange', linestyle='--', alpha=0.5)
ax.grid(axis='x', linestyle='--', alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
cons_path = os.path.join(OUTPUT_DIR, '09_konsensus_analizi.png')
plt.savefig(cons_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'   ✅ {cons_path}')

print('\n📊 Isı haritası hazırlanıyor...')
top_feats    = vote_ser[vote_ser >= 2].index.tolist()
heatmap_df   = shap_df.loc[top_feats].copy()
heatmap_norm = heatmap_df.div(heatmap_df.max(axis=1), axis=0)

fig, ax = plt.subplots(figsize=(12, max(6, len(top_feats) * 0.45)))
im = ax.imshow(heatmap_norm.values, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
ax.set_xticks(range(len(model_list)))
ax.set_xticklabels(model_list, rotation=30, ha='right', fontsize=10)
ax.set_yticks(range(len(top_feats)))
ax.set_yticklabels(top_feats, fontsize=9)
for i, feat in enumerate(top_feats):
    for j, mname in enumerate(model_list):
        val = heatmap_norm.loc[feat, mname]
        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                fontsize=7.5, color='black' if val < 0.7 else 'white')
plt.colorbar(im, ax=ax, label='Normalize SHAP Önemi (0–1)', shrink=0.8)
ax.set_title('SHAP Isı Haritası — Model × Özellik',
             fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
heat_path = os.path.join(OUTPUT_DIR, '10_isi_haritasi.png')
plt.savefig(heat_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'   ✅ {heat_path}')

csv_df            = pd.DataFrame({m: rankings[m] for m in model_list})
csv_df.index.name = 'Feature'
csv_df.to_csv(os.path.join(OUTPUT_DIR, 'shap_feature_rankings.csv'))

vote_df = pd.DataFrame({
    'Feature'    : vote_ser.index,
    'Oy_Sayisi'  : vote_ser.values,
    'Oy_Yuzdesi' : (vote_ser.values / n_models_total * 100).round(1)
})
vote_df.to_csv(os.path.join(OUTPUT_DIR, 'konsensus_ozet.csv'), index=False)
print(f'   ✅ CSV\'ler kaydedildi')

print('\n' + '='*60)
print('  SHAP ANALİZİ TAMAMLANDI ✅')
print('='*60)
print(f'\n  {"Model":<22} | Top 3 SHAP Özelliği')
print('  ' + '-'*55)
for mname, ser in rankings.items():
    top3 = ', '.join(ser.head(3).index.tolist())
    print(f'  {mname:<22} | {top3}')

print(f'\n  {"─"*55}')
print(f'  🏆 KONSENSÜS:')
print(f'  {"─"*55}')
for feat, cnt in vote_ser.items():
    bar  = '█' * cnt + '░' * (n_models_total - cnt)
    mark = ' ← HEPSİ SEÇTİ' if cnt == n_models_total else ''
    print(f'  {feat:<30} {bar}  {cnt}/{n_models_total}{mark}')

print(f'\n  Üretilen dosyalar:')
for f in sorted(os.listdir(OUTPUT_DIR)):
    print(f'    📄 {f}')
print('='*60 + '\n')
