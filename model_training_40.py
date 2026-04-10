"""
Model Eğitim ve Karşılaştırma Pipeline'ı
=========================================
Kötücül JavaScript Tespiti — Makine Öğrenmesi Projesi

Veri      : fs_output/features_selected.csv  (40 seçilmiş feature, 52.830 örnek)
Split     : 'split' kolonu kullanılır (feature selection'da oluşturuldu)
           Yoksa stratified 80/20 otomatik uygulanır

Modeller  :
  ML  → Logistic Regression, Naive Bayes, KNN, SVM,
         Decision Tree, Random Forest, XGBoost
  DL  → MLP, CNN (1D), LSTM, BiLSTM, GRU

Metrikler : Accuracy, Precision, Recall, F1, ROC-AUC, Süre(s)

Çıktılar (results_40/ klasörüne):
  model_sonuclari_40.csv
  01_model_karsilastirma.png   — 5 metrik bar chart (tüm modeller)
  02_f1_karsilastirma.png      — ML vs DL F1 karşılaştırması
  03_roc_curves.png            — ML ve DL ROC eğrileri
  04_confusion_matrix.png      — en iyi 4 modelin confusion matrix'i
  05_ml_vs_dl.png              — ML vs DL ortalama performans
  06_sure_karsilastirma.png    — eğitim süreleri

Not: 114 feature sonuçlarıyla karşılaştırma için
     results/ klasöründeki model_sonuclari.csv gereklidir.

Gereksinimler:
  pip install tensorflow xgboost scikit-learn pandas numpy matplotlib seaborn
"""

import os, warnings, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix, roc_curve)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Dropout, Conv1D, GlobalAveragePooling1D,
                                      LSTM, Bidirectional, GRU,
                                      BatchNormalization, Reshape)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# ── Ayarlar ───────────────────────────────────────────────────────────────────
INPUT_CSV  = 'features_selected.csv'   # 40 seçilmiş feature
RESULTS    = 'results_40'
SEED       = 42
EPOCHS     = 50
BATCH_SIZE = 256
os.makedirs(RESULTS, exist_ok=True)

ML_NAMES = ['Logistic Regression', 'Naive Bayes', 'KNN', 'SVM',
            'Decision Tree', 'Random Forest', 'XGBoost']

MODEL_COLORS = {
    'Logistic Regression': '#3498db', 'Naive Bayes':   '#2ecc71',
    'KNN':                 '#e74c3c', 'SVM':           '#f39c12',
    'Decision Tree':       '#9b59b6', 'Random Forest': '#1abc9c',
    'XGBoost':             '#e67e22', 'MLP':           '#e91e63',
    'CNN':                 '#00bcd4', 'LSTM':          '#8bc34a',
    'BiLSTM':              '#ff5722', 'GRU':           '#607d8b',
}

print('=' * 62)
print('  JS Kötücül Kod Tespiti — 12 Model (40 Seçilmiş Feature)')
print('=' * 62)

# ══════════════════════════════════════════════════════════════════════════════
# 1. VERİ YÜKLEME & SPLIT
# ══════════════════════════════════════════════════════════════════════════════
df = pd.read_csv(INPUT_CSV)

if 'split' in df.columns:
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    test_df  = df[df['split'] == 'test'].reset_index(drop=True)
    print(f'\n  Split kolonu bulundu — feature selection split\'i kullanılıyor.')
else:
    print(f'\n  Split kolonu yok → stratified 80/20 uygulanıyor...')
    train_idx, test_idx = train_test_split(
        df.index, test_size=0.2, random_state=SEED, stratify=df['label'])
    train_df = df.loc[train_idx].reset_index(drop=True)
    test_df  = df.loc[test_idx].reset_index(drop=True)

drop_cols = [c for c in ['filename', 'label', 'split'] if c in df.columns]
X_train   = train_df.drop(columns=drop_cols).select_dtypes(include=[np.number])
y_train   = train_df['label'].values
X_test    = test_df.drop(columns=drop_cols).select_dtypes(include=[np.number])
y_test    = test_df['label'].values

print(f'  Toplam   : {len(df):,}')
print(f'  Train    : {len(X_train):,}  '
      f'(Benign: {(y_train==0).sum():,} | Malicious: {(y_train==1).sum():,})')
print(f'  Test     : {len(X_test):,}  '
      f'(Benign: {(y_test==0).sum():,}  | Malicious: {(y_test==1).sum():,})')
print(f'  Feature  : {X_train.shape[1]}')

# Ölçekleme — sadece train'e fit
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
n_features = X_train_sc.shape[1]

# Sınıf ağırlıkları
cw        = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
cw_dict   = {0: float(cw[0]), 1: float(cw[1])}
scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
print(f'  Sınıf ağırlıkları → Benign: {cw[0]:.3f} | Malicious: {cw[1]:.3f}')

# ══════════════════════════════════════════════════════════════════════════════
# 2. YARDIMCI FONKSİYONLAR
# ══════════════════════════════════════════════════════════════════════════════
results       = []
conf_matrices = {}
roc_data      = {}

def record(name, y_pred, y_prob, elapsed):
    """Metrikleri hesapla, kaydet ve yazdır."""
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_prob)
    print(f'  ✅ {name:<25} Acc:{acc:.4f}  P:{prec:.4f}  R:{rec:.4f}  '
          f'F1:{f1:.4f}  AUC:{auc:.4f}  ({elapsed:.1f}s)')
    results.append({'Model': name, 'Accuracy': acc, 'Precision': prec,
                    'Recall': rec, 'F1': f1, 'ROC-AUC': auc,
                    'Süre(s)': round(elapsed, 1)})
    conf_matrices[name] = confusion_matrix(y_test, y_pred)
    fpr, tpr, _         = roc_curve(y_test, y_prob)
    roc_data[name]      = (fpr, tpr, auc)

def dl_train(name, model, patience=7):
    """DL modelini eğit, değerlendir, belleği temizle."""
    es  = EarlyStopping(monitor='val_loss', patience=patience,
                        restore_best_weights=True)
    rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=0)
    t   = time.time()
    h   = model.fit(X_train_sc, y_train,
                    epochs=EPOCHS, batch_size=BATCH_SIZE,
                    validation_split=0.1,
                    callbacks=[es, rlr],
                    class_weight=cw_dict,
                    verbose=0)
    elapsed = time.time() - t
    y_prob  = model.predict(X_test_sc, verbose=0).flatten()
    y_pred  = (y_prob >= 0.5).astype(int)
    print(f'     Epoch: {len(h.history["loss"])}')
    record(name, y_pred, y_prob, elapsed)
    tf.keras.backend.clear_session()

# ══════════════════════════════════════════════════════════════════════════════
# 3. ML MODELLERİ
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '─' * 62)
print('  📊 GELENEKSEL MAKİNE ÖĞRENMESİ')
print('─' * 62)

# [1] Logistic Regression
print('\n[1/12] Logistic Regression...')
t  = time.time()
lr = LogisticRegression(max_iter=1000, class_weight='balanced',
                         random_state=SEED, n_jobs=-1)
lr.fit(X_train_sc, y_train)
record('Logistic Regression', lr.predict(X_test_sc),
       lr.predict_proba(X_test_sc)[:, 1], time.time() - t)

# [2] Naive Bayes — class_weight desteklemiyor, prior dengeli ayarlandı
print('[2/12] Naive Bayes...')
t  = time.time()
nb = GaussianNB(priors=[0.5, 0.5])
nb.fit(X_train_sc, y_train)
record('Naive Bayes', nb.predict(X_test_sc),
       nb.predict_proba(X_test_sc)[:, 1], time.time() - t)

# [3] KNN — class_weight yok, weights='distance' ile yakın komşulara daha fazla ağırlık
print('[3/12] KNN  (weights=distance)...')
t   = time.time()
knn = KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)
knn.fit(X_train_sc, y_train)
record('KNN', knn.predict(X_test_sc),
       knn.predict_proba(X_test_sc)[:, 1], time.time() - t)

# [4] SVM
print('[4/12] SVM  (uzun sürebilir)...')
t   = time.time()
svm = SVC(kernel='rbf', class_weight='balanced',
           probability=True, random_state=SEED)
svm.fit(X_train_sc, y_train)
record('SVM', svm.predict(X_test_sc),
       svm.predict_proba(X_test_sc)[:, 1], time.time() - t)

# [5] Decision Tree
print('[5/12] Decision Tree...')
t  = time.time()
dt = DecisionTreeClassifier(class_weight='balanced',
                             random_state=SEED, max_depth=20)
dt.fit(X_train_sc, y_train)
record('Decision Tree', dt.predict(X_test_sc),
       dt.predict_proba(X_test_sc)[:, 1], time.time() - t)

# [6] Random Forest
print('[6/12] Random Forest...')
t  = time.time()
rf = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                              random_state=SEED, n_jobs=-1)
rf.fit(X_train_sc, y_train)
record('Random Forest', rf.predict(X_test_sc),
       rf.predict_proba(X_test_sc)[:, 1], time.time() - t)

# [7] XGBoost — xgb_model isim çakışmasını önlemek için
print('[7/12] XGBoost...')
t         = time.time()
xgb_model = XGBClassifier(n_estimators=200, scale_pos_weight=scale_pos,
                            random_state=SEED, n_jobs=-1,
                            eval_metric='logloss', verbosity=0)
xgb_model.fit(X_train_sc, y_train)
record('XGBoost', xgb_model.predict(X_test_sc),
       xgb_model.predict_proba(X_test_sc)[:, 1], time.time() - t)

# ══════════════════════════════════════════════════════════════════════════════
# 4. DL MODELLERİ  (tümü aynı patience=7, ReduceLROnPlateau ile)
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '─' * 62)
print('  🧠 DERİN ÖĞRENME')
print('─' * 62)

# [8] MLP
print('\n[8/12] MLP...')
mlp = Sequential([
    Dense(256, activation='relu', input_shape=(n_features,)),
    BatchNormalization(), Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(), Dropout(0.3),
    Dense(64,  activation='relu'), Dropout(0.2),
    Dense(1,   activation='sigmoid')
])
mlp.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
dl_train('MLP', mlp)

# [9] CNN
print('\n[9/12] CNN...')
cnn = Sequential([
    Reshape((n_features, 1), input_shape=(n_features,)),
    Conv1D(128, 3, activation='relu', padding='same'), BatchNormalization(),
    Conv1D(64,  3, activation='relu', padding='same'), BatchNormalization(),
    GlobalAveragePooling1D(),
    Dense(128, activation='relu'), Dropout(0.3),
    Dense(64,  activation='relu'), Dropout(0.2),
    Dense(1,   activation='sigmoid')
])
cnn.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
dl_train('CNN', cnn)

# [10] LSTM
print('\n[10/12] LSTM...')
lstm_m = Sequential([
    Reshape((n_features, 1), input_shape=(n_features,)),
    LSTM(128, return_sequences=True), Dropout(0.3),
    LSTM(64), Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1,  activation='sigmoid')
])
lstm_m.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
dl_train('LSTM', lstm_m)

# [11] BiLSTM
print('\n[11/12] BiLSTM...')
bilstm = Sequential([
    Reshape((n_features, 1), input_shape=(n_features,)),
    Bidirectional(LSTM(128, return_sequences=True)), Dropout(0.3),
    Bidirectional(LSTM(64)), Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1,  activation='sigmoid')
])
bilstm.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
dl_train('BiLSTM', bilstm)

# [12] GRU
print('\n[12/12] GRU...')
gru_m = Sequential([
    Reshape((n_features, 1), input_shape=(n_features,)),
    GRU(128, return_sequences=True), Dropout(0.3),
    GRU(64), Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1,  activation='sigmoid')
])
gru_m.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
dl_train('GRU', gru_m)

# ══════════════════════════════════════════════════════════════════════════════
# 5. SONUÇ TABLOSU & CSV
# ══════════════════════════════════════════════════════════════════════════════
df_res  = pd.DataFrame(results).sort_values('F1', ascending=False).reset_index(drop=True)
df_res.to_csv(f'{RESULTS}/model_sonuclari_40.csv', index=False)

metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
ml_df   = df_res[df_res['Model'].isin(ML_NAMES)]
dl_df   = df_res[~df_res['Model'].isin(ML_NAMES)]

print('\n' + '=' * 62)
print("  📊 SONUÇ TABLOSU — 40 Feature  (F1'e göre sıralı)")
print('=' * 62)
print(df_res[['Model', 'Accuracy', 'Precision', 'Recall',
              'F1', 'ROC-AUC', 'Süre(s)']
            ].to_string(index=False, float_format='{:.4f}'.format))

# ══════════════════════════════════════════════════════════════════════════════
# 6. GRAFİKLER
# ══════════════════════════════════════════════════════════════════════════════

# ── Grafik 1: 5 metrik bar chart ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(26, 7))
for ax, metric in zip(axes, metrics):
    vals  = df_res[metric].values
    names = df_res['Model'].values
    cols  = [MODEL_COLORS.get(n, '#607d8b') for n in names]
    bars  = ax.barh(names, vals, color=cols, edgecolor='white')
    ax.set_xlim(max(0, min(vals) - 0.05), 1.06)
    ax.set_title(metric, fontsize=12, fontweight='bold')
    for bar, val in zip(bars, vals):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=7.5)
    ax.axvline(x=0.95, color='red', linestyle='--', alpha=0.4, linewidth=0.8)
    ax.spines[['top', 'right']].set_visible(False)
plt.suptitle('12 Model — Tüm Metrikler  (40 Seçilmiş Feature)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{RESULTS}/01_model_karsilastirma.png', dpi=150, bbox_inches='tight')
plt.close()
print('\n✅ 01_model_karsilastirma.png')

# ── Grafik 2: ML vs DL F1 ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 7))
bar_cols = ['#2980b9' if m in ML_NAMES else '#c0392b' for m in df_res['Model']]
bars     = ax.barh(df_res['Model'], df_res['F1'],
                   color=bar_cols, edgecolor='white', height=0.6)
for bar, val in zip(bars, df_res['F1']):
    ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
            f'{val:.4f}', va='center', fontsize=10, fontweight='bold')
ax.set_xlim(max(0, df_res['F1'].min() - 0.05), 1.06)
ax.set_xlabel('F1 Skoru', fontsize=12)
ax.set_title('Model Karşılaştırması — F1 Skoru\n(Mavi: ML  |  Kırmızı: DL)  [40 Feature]',
             fontsize=13, fontweight='bold')
ax.axvline(x=0.95, color='orange', linestyle='--', alpha=0.5)
ax.legend(handles=[mpatches.Patch(color='#2980b9', label='Geleneksel ML'),
                   mpatches.Patch(color='#c0392b', label='Derin Öğrenme'),
                   mpatches.Patch(color='orange',  label='0.95 eşiği', alpha=0.5)],
          fontsize=10)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig(f'{RESULTS}/02_f1_karsilastirma.png', dpi=150, bbox_inches='tight')
plt.close()
print('✅ 02_f1_karsilastirma.png')

# ── Grafik 3: ROC eğrileri ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
for ax, grup, baslik in zip(
        axes,
        [ML_NAMES, ['MLP', 'CNN', 'LSTM', 'BiLSTM', 'GRU']],
        ['Geleneksel ML — ROC Eğrileri', 'Derin Öğrenme — ROC Eğrileri']):
    for name in grup:
        if name not in roc_data:
            continue
        fpr, tpr, auc = roc_data[name]
        ax.plot(fpr, tpr, label=f'{name}  (AUC={auc:.4f})',
                color=MODEL_COLORS.get(name, '#607d8b'), linewidth=1.8)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title(baslik, fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.spines[['top', 'right']].set_visible(False)
plt.suptitle('ROC Eğrileri — 40 Seçilmiş Feature', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{RESULTS}/03_roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print('✅ 03_roc_curves.png')

# ── Grafik 4: En iyi 4 model Confusion Matrix ─────────────────────────────────
best4     = df_res.head(4)['Model'].tolist()
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for ax, name in zip(axes, best4):
    cm  = conf_matrices[name]
    pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                xticklabels=['Benign', 'Malicious'],
                yticklabels=['Benign', 'Malicious'])
    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.78, f'({pct[i,j]:.1f}%)',
                    ha='center', va='center', fontsize=8, color='gray')
    f1v = df_res[df_res['Model'] == name]['F1'].values[0]
    ax.set_title(f'{name}\nF1={f1v:.4f}', fontsize=11, fontweight='bold')
    ax.set_ylabel('Gerçek')
    ax.set_xlabel('Tahmin')
plt.suptitle('En İyi 4 Model — Confusion Matrix  [40 Feature]',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{RESULTS}/04_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print('✅ 04_confusion_matrix.png')

# ── Grafik 5: ML vs DL ortalama ───────────────────────────────────────────────
ml_avg = ml_df[metrics].mean()
dl_avg = dl_df[metrics].mean()
fig, ax = plt.subplots(figsize=(10, 6))
x, w = np.arange(len(metrics)), 0.35
ax.bar(x - w/2, ml_avg, w, label='ML Ortalaması', color='#2980b9', alpha=0.85)
ax.bar(x + w/2, dl_avg, w, label='DL Ortalaması', color='#c0392b', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=12)
ax.set_ylim(0, 1.12)
ax.set_ylabel('Ortalama Skor', fontsize=12)
ax.set_title('Geleneksel ML vs Derin Öğrenme — Ortalama Performans  [40 Feature]',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=11)
ax.spines[['top', 'right']].set_visible(False)
for i, (mv, dv) in enumerate(zip(ml_avg, dl_avg)):
    ax.text(i - w/2, mv + 0.01, f'{mv:.4f}', ha='center', fontsize=9, fontweight='bold')
    ax.text(i + w/2, dv + 0.01, f'{dv:.4f}', ha='center', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{RESULTS}/05_ml_vs_dl.png', dpi=150, bbox_inches='tight')
plt.close()
print('✅ 05_ml_vs_dl.png')

# ── Grafik 6: Eğitim süreleri (log scale) ─────────────────────────────────────
sure_df  = df_res.sort_values('Süre(s)', ascending=True)
bar_cols = ['#2980b9' if m in ML_NAMES else '#c0392b' for m in sure_df['Model']]
fig, ax  = plt.subplots(figsize=(12, 6))
bars     = ax.barh(sure_df['Model'], sure_df['Süre(s)'],
                   color=bar_cols, edgecolor='white', height=0.6)
for bar, val in zip(bars, sure_df['Süre(s)']):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f'{val:.1f}s', va='center', fontsize=10, fontweight='bold')
ax.set_xlabel('Eğitim Süresi (saniye)', fontsize=12)
ax.set_title('Model Eğitim Süreleri  [40 Seçilmiş Feature]', fontsize=13, fontweight='bold')
ax.legend(handles=[mpatches.Patch(color='#2980b9', label='Geleneksel ML'),
                   mpatches.Patch(color='#c0392b', label='Derin Öğrenme')], fontsize=10)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig(f'{RESULTS}/06_sure_karsilastirma.png', dpi=150, bbox_inches='tight')
plt.close()
print('✅ 06_sure_karsilastirma.png')

# ══════════════════════════════════════════════════════════════════════════════
# 7. 114 vs 40 FEATURE KARŞILAŞTIRMA GRAFİĞİ
# ══════════════════════════════════════════════════════════════════════════════
csv_114 = 'results_114/model_sonuclari_114.csv'
if os.path.exists(csv_114):
    df_114 = pd.read_csv(csv_114)
    df_40  = df_res.copy()

    # Ortak modeller
    common_models = df_114['Model'].tolist()
    df_40_aligned  = df_40.set_index('Model').reindex(common_models)
    df_114_aligned = df_114.set_index('Model').reindex(common_models)

    compare_metrics = ['F1', 'ROC-AUC', 'Accuracy']
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    for ax, metric in zip(axes, compare_metrics):
        y_pos  = np.arange(len(common_models))
        w      = 0.35
        vals_114 = df_114_aligned[metric].values
        vals_40  = df_40_aligned[metric].values

        bars1 = ax.barh(y_pos - w/2, vals_114, w,
                        label='114 Feature (ham)', color='#546E7A',
                        edgecolor='white', alpha=0.85)
        bars2 = ax.barh(y_pos + w/2, vals_40,  w,
                        label='40 Feature (seçilmiş)', color='#FFD700',
                        edgecolor='white', alpha=0.85)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(common_models, fontsize=9)
        ax.set_xlabel(metric, fontsize=11)
        ax.set_title(f'{metric} Karşılaştırması', fontsize=12, fontweight='bold')

        mn = max(0, min(np.nanmin(vals_114), np.nanmin(vals_40)) - 0.05)
        ax.set_xlim(mn, 1.06)

        for bar, val in zip(bars2, vals_40):
            if not np.isnan(val):
                ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                        f'{val:.4f}', va='center', fontsize=7)

        ax.axvline(x=0.99, color='red', linestyle='--', alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(fontsize=9, loc='lower right')

    plt.suptitle('Özellik Seçiminin Etkisi — 114 Ham vs 40 Seçilmiş Feature',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{RESULTS}/07_114_vs_40_karsilastirma.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('✅ 07_114_vs_40_karsilastirma.png')

    # Özet tablo: ortalama fark
    print(f'\n{"─"*62}')
    print('  📊 114 vs 40 FEATURE KARŞILAŞTIRMA ÖZETİ')
    print(f'{"─"*62}')
    for metric in ['F1', 'ROC-AUC', 'Accuracy']:
        avg_114 = df_114_aligned[metric].mean()
        avg_40  = df_40_aligned[metric].mean()
        diff    = avg_40 - avg_114
        sign    = '+' if diff >= 0 else ''
        print(f'  {metric:<10} 114 feat ort: {avg_114:.4f}  |  '
              f'40 feat ort: {avg_40:.4f}  |  fark: {sign}{diff:.4f}')
else:
    print(f'\n  ⚠️  {csv_114} bulunamadı — 114 vs 40 karşılaştırma grafiği atlandı.')
    print(f'      Önce 114 feature scriptini çalıştır.')

# ══════════════════════════════════════════════════════════════════════════════
# 8. ÖZET
# ══════════════════════════════════════════════════════════════════════════════
best    = df_res.iloc[0]
best_ml = ml_df.iloc[0]
best_dl = dl_df.iloc[0]

print(f"""
{'='*62}
  ✅ MODEL EĞİTİMİ TAMAMLANDI  (40 Seçilmiş Feature)
{'='*62}
  En iyi model  : {best['Model']}
  F1 Skoru      : {best['F1']:.4f}
  Accuracy      : {best['Accuracy']:.4f}
  ROC-AUC       : {best['ROC-AUC']:.4f}
  Eğitim süresi : {best['Süre(s)']:.1f}s

  En iyi ML     : {best_ml['Model']:<22} F1={best_ml['F1']:.4f}
  En iyi DL     : {best_dl['Model']:<22} F1={best_dl['F1']:.4f}

  ML Ortalama F1 : {ml_df['F1'].mean():.4f}
  DL Ortalama F1 : {dl_df['F1'].mean():.4f}

  Çıktılar → {RESULTS}/
  ✅ model_sonuclari_40.csv
  ✅ 01_model_karsilastirma.png
  ✅ 02_f1_karsilastirma.png
  ✅ 03_roc_curves.png
  ✅ 04_confusion_matrix.png
  ✅ 05_ml_vs_dl.png
  ✅ 06_sure_karsilastirma.png
  ✅ 07_114_vs_40_karsilastirma.png  (114 feature CSV mevcutsa)
{'='*62}
""")