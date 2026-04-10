"""
Özellik grupları:
  A. Yapısal     
  B. Karakter     
  C. Gizleme      
  D. API          
  E. Kod yapısı   
  F. Ağ / URL     

"""

import re, os, sys, math, json, time, zipfile, collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import signal

SCRIPT_DIR    = Path(__file__).parent
BENIGN_ZIP    = SCRIPT_DIR / 'benign.zip'
MALICIOUS_ZIP = SCRIPT_DIR / 'malicious.zip'
OUTPUT_DIR    = SCRIPT_DIR / 'output'
OUTPUT_CSV    = OUTPUT_DIR / 'features.csv'
CHECKPOINT    = OUTPUT_DIR / 'checkpoint.json'
GRAPHS_DIR    = OUTPUT_DIR / 'graphs'

OUTPUT_DIR.mkdir(exist_ok=True)
GRAPHS_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("  JS Kötücül Kod Tespiti — Feature Extraction")
print("=" * 60)
print(f"  benign.zip    : {'✅ BULUNDU' if BENIGN_ZIP.exists() else '❌ BULUNAMADI'}")
print(f"  malicious.zip : {'✅ BULUNDU' if MALICIOUS_ZIP.exists() else '❌ BULUNAMADI'}")

if not BENIGN_ZIP.exists() or not MALICIOUS_ZIP.exists():
    print(f"\n  Lütfen zip dosyalarını şu klasöre koy:\n  {SCRIPT_DIR}")
    sys.exit(1)

def lexical_features(code):
    f = {}
    f['file_length']           = len(code)
    f['num_lines']             = code.count('\n') + 1
    f['avg_line_length']       = len(code) / max(1, f['num_lines'])
    f['max_line_length']       = max((len(l) for l in code.split('\n')), default=0)
    n = max(len(code), 1)
    f['ratio_alpha']           = sum(c.isalpha()  for c in code) / n
    f['ratio_digit']           = sum(c.isdigit()  for c in code) / n
    f['ratio_space']           = sum(c.isspace()  for c in code) / n
    f['ratio_special']         = sum(not c.isalnum() and not c.isspace() for c in code) / n
    f['ratio_uppercase']       = sum(c.isupper()  for c in code) / n
    f['count_semicolon']       = code.count(';')
    f['count_paren']           = code.count('(') + code.count(')')
    f['count_bracket']         = code.count('{') + code.count('}')
    f['count_square']          = code.count('[') + code.count(']')
    f['count_plus']            = code.count('+')
    f['count_pipe']            = code.count('|')
    f['count_percent']         = code.count('%')
    f['count_single_quote']    = code.count("'")
    f['count_double_quote']    = code.count('"')
    f['count_backtick']        = code.count('`')
    f['count_string_literals'] = len(re.findall(r"""(["'])(?:(?!\1)[^\\]|\\.)*\1""", code))
    f['count_line_comments']   = len(re.findall(r'//[^\n]*', code))
    f['count_block_comments']  = len(re.findall(r'/\*.*?\*/', code, re.DOTALL))
    f['unique_chars']          = len(set(code))
    return f

def shannon_entropy(s):
    if not s: return 0.0
    cnt = collections.Counter(s)
    t   = len(s)
    return -sum((v/t)*math.log2(v/t) for v in cnt.values())

def entropy_features(code):
    f = {}
    f['entropy_full']         = shannon_entropy(code)
    strings                   = re.findall(r"""(["'])(?:(?!\1)[^\\]|\\.)*\1""", code)
    f['entropy_strings']      = shannon_entropy(''.join(strings))
    f['num_long_strings']     = sum(1 for s in strings if len(s) > 100)
    f['max_string_length']    = max((len(s) for s in strings), default=0)
    f['avg_string_length']    = float(np.mean([len(s) for s in strings])) if strings else 0
    f['count_hex_escape']     = len(re.findall(r'(\\x[0-9a-fA-F]{2}){3,}', code))
    f['count_unicode_escape'] = len(re.findall(r'\\u[0-9a-fA-F]{4}', code))
    b64                       = re.findall(r'[A-Za-z0-9+/=]{40,}', code)
    f['count_b64_like']       = len(b64)
    f['max_b64_length']       = max((len(s) for s in b64), default=0)
    return f

def syntax_features(code):
    f = {}
    f['count_function_decl']   = len(re.findall(r'\bfunction\s+\w+\s*\(', code))
    f['count_function_expr']   = len(re.findall(r'\bfunction\s*\(', code))
    f['count_arrow_func']      = len(re.findall(r'=>', code))
    f['count_total_functions'] = f['count_function_decl'] + f['count_function_expr'] + f['count_arrow_func']
    f['count_for']             = len(re.findall(r'\bfor\s*\(', code))
    f['count_while']           = len(re.findall(r'\bwhile\s*\(', code))
    f['count_if']              = len(re.findall(r'\bif\s*\(', code))
    f['count_try_catch']       = len(re.findall(r'\btry\s*\{', code))
    f['count_switch']          = len(re.findall(r'\bswitch\s*\(', code))
    f['count_var']             = len(re.findall(r'\bvar\b', code))
    f['count_let']             = len(re.findall(r'\blet\b', code))
    f['count_const']           = len(re.findall(r'\bconst\b', code))
    f['count_prototype']       = len(re.findall(r'\.prototype\b', code))
    f['count_this']            = len(re.findall(r'\bthis\b', code))
    depth = max_depth = 0
    for ch in code:
        if ch == '{': depth += 1; max_depth = max(max_depth, depth)
        elif ch == '}': depth = max(0, depth - 1)
    f['max_nesting_depth']    = max_depth
    f['count_return']         = len(re.findall(r'\breturn\b', code))
    f['count_ternary']        = code.count('?')
    return f

DANGEROUS_APIS = {
    'eval':                r'\beval\s*\(',
    'Function':            r'\bFunction\s*\(',
    'setTimeout_code':     r'\bsetTimeout\s*\(\s*["\']',
    'setInterval_code':    r'\bsetInterval\s*\(\s*["\']',
    'execScript':          r'\bexecScript\s*\(',
    'innerHTML':           r'\.innerHTML\s*[=+]',
    'outerHTML':           r'\.outerHTML\s*[=+]',
    'document_write':      r'\bdocument\.write\s*\(',
    'insertAdjacentHTML':  r'\.insertAdjacentHTML\s*\(',
    'createElement_script':r'createElement\s*\(\s*["\']script["\']',
    'src_assign':          r'\.src\s*=',
    'XMLHttpRequest':      r'\bXMLHttpRequest\b',
    'fetch_api':           r'\bfetch\s*\(',
    'axios':               r'\baxios\b',
    'WebSocket':           r'\bWebSocket\s*\(',
    'atob':                r'\batob\s*\(',
    'btoa':                r'\bbtoa\s*\(',
    'unescape':            r'\bunescape\s*\(',
    'decodeURIComponent':  r'\bdecodeURIComponent\s*\(',
    'fromCharCode':        r'\bfromCharCode\s*\(',
    'charCodeAt':          r'\bcharCodeAt\s*\(',
    'String_fromCharCode': r'String\.fromCharCode\s*\(',
    'join_empty':          r'\.join\s*\(\s*["\']["\']',
    'split_empty':         r'\.split\s*\(\s*["\']["\']',
    'replace_regex':       r'\.replace\s*\(\s*/',
    '__proto__':           r'__proto__',
    'constructor_proto':   r'\.constructor\s*\[',
    'iframe':              r'<\s*iframe',
    'onload_attr':         r'onload\s*=',
    'document_cookie':     r'document\.cookie',
    'localStorage':        r'\blocalStorage\b',
    'sessionStorage':      r'\bsessionStorage\b',
    'location_href':       r'\blocation\s*\.href\s*=',
    'location_replace':    r'\blocation\.replace\s*\(',
    'location_assign':     r'\blocation\.assign\s*\(',
    'require_child':       r'\brequire\s*\(\s*["\']child_process',
    'require_fs':          r'\brequire\s*\(\s*["\']fs["\']',
    'process_env':         r'\bprocess\.env\b',
    'exec_spawn':          r'\b(exec|spawn)\s*\(',
    'CoinHive':            r'\bCoinHive\b',
    'wasm':                r'\bWebAssembly\b',
}

def api_features(code):
    f = {}
    cnt_nonzero = 0
    for name, pattern in DANGEROUS_APIS.items():
        cnt = len(re.findall(pattern, code, re.IGNORECASE))
        f[f'api_{name}'] = cnt
        if cnt > 0: cnt_nonzero += 1
    f['dangerous_api_total']     = sum(f[f'api_{k}'] for k in DANGEROUS_APIS)
    f['dangerous_api_diversity'] = cnt_nonzero
    return f

def obfuscation_features(code):
    f = {}
    f['hex_encoded_chars']     = len(re.findall(r'\\x[0-9a-fA-F]{2}', code))
    f['count_number_arrays']   = len(re.findall(r'\[(\s*\d+\s*,\s*){4,}\d+\s*\]', code))
    long_lines                 = [l for l in code.split('\n') if len(l) > 500]
    f['count_very_long_lines'] = len(long_lines)
    f['max_single_line']       = max((len(l) for l in code.split('\n')), default=0)
    f['count_str_concat']      = len(re.findall(r'["\'][^"\']*["\'](\s*\+\s*["\'][^"\']*["\']){3,}', code))
    f['count_hex_numbers']     = len(re.findall(r'\b0x[0-9a-fA-F]+\b', code))
    f['count_octal_numbers']   = len(re.findall(r'\b0[0-7]{2,}\b', code))
    f['count_bracket_access']  = len(re.findall(r'\w+\s*\[\s*["\']', code))
    f['count_window_bracket']  = len(re.findall(r'window\s*\[\s*["\']', code))
    f['count_encoded_newline'] = code.count('\\n') + code.count('\\t')
    f['count_regex']           = len(re.findall(r'/[^/\n]{5,}/[gimsuy]*', code))
    short_vars                 = re.findall(r'\bvar\s+([a-zA-Z_$]{1,2})\b', code)
    f['count_short_varnames']  = len(short_vars)
    f['unique_short_varnames'] = len(set(short_vars))
    return f

def network_features(code):
    f = {}
    urls                      = re.findall(r'https?://[^\s"\'<>]+', code)
    f['count_urls']           = len(urls)
    f['count_http_urls']      = sum(1 for u in urls if u.startswith('http://'))
    f['count_https_urls']     = sum(1 for u in urls if u.startswith('https://'))
    f['count_ip_address']     = len(re.findall(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', code))
    f['count_port_pattern']   = len(re.findall(r':\d{4,5}(?:[/\s"\')])', code))
    f['count_data_url']       = len(re.findall(r'data:[a-z]+/[a-z]+;base64,', code))
    f['count_suspicious_tld'] = len(re.findall(r'\.(xyz|tk|ml|ga|cf|gq|pw|top|click|download)\b', code, re.IGNORECASE))
    return f

def extract_features(code, label=-1, filename=''):
    features = {'filename': filename, 'label': label}
    try:
        if len(code) > 500_000:
            code = code[:500_000]
        features.update(lexical_features(code))
        features.update(entropy_features(code))
        features.update(syntax_features(code))
        features.update(api_features(code))
        features.update(obfuscation_features(code))
        features.update(network_features(code))
        n = max(len(code), 1)
        features['api_density']  = features['dangerous_api_total'] / n * 1000
        features['obfusc_score'] = (
            features['count_hex_escape']      * 2 +
            features['count_unicode_escape']  * 1 +
            features['count_b64_like']        * 3 +
            features['count_very_long_lines'] * 2 +
            features['count_str_concat']      * 2 +
            features['count_window_bracket']  * 3 +
            features['api_eval']              * 5 +
            features['api_atob']              * 3 +
            features['api_fromCharCode']      * 3
        )
    except Exception as e:
        features['_error'] = str(e)
    return features

def save_checkpoint(label_name, last_index):
    with open(CHECKPOINT, 'w') as f:
        json.dump({'label_name': label_name, 'last_index': last_index}, f)

def load_checkpoint():
    if CHECKPOINT.exists():
        with open(CHECKPOINT) as f:
            return json.load(f)
    return None

def clear_checkpoint():
    if CHECKPOINT.exists():
        CHECKPOINT.unlink()

def process_zip(zip_path, label, label_name, start_from=0):
    BATCH_SIZE = 500
    batch_csv  = OUTPUT_DIR / f'batch_{label_name}.csv'
    records    = []
    t_start    = time.time()

    with zipfile.ZipFile(zip_path, 'r') as zf:
        js_files = sorted([
            f for f in zf.namelist()
            if f.endswith('.js') and not f.startswith('__MACOSX')
        ])
        total = len(js_files)

        if start_from > 0:
            print(f"  ⏩ {start_from} dosya zaten işlendi, devam ediliyor...")
        print(f"  📂 {label_name.upper()} — toplam {total} dosya, "
              f"{total - start_from} kaldı\n")

        for i, fname in enumerate(js_files):
            if i < start_from:
                continue

            try:
                with zf.open(fname) as fh:
                    raw = fh.read()
                    try:    code = raw.decode('utf-8')
                    except: code = raw.decode('latin-1', errors='replace')
                    if len(code.strip()) < 10:
                        continue
                    records.append(extract_features(code, label=label, filename=fname))
            except:
                pass

            done = i + 1
            if done % BATCH_SIZE == 0 or done == total:
                elapsed   = time.time() - t_start
                speed     = (done - start_from) / max(elapsed, 1)
                remaining = (total - done) / max(speed, 0.1)
                pct       = 100 * done / total

                print(f"  [{label_name}] {done:>6}/{total} "
                      f"({pct:5.1f}%) | "
                      f"{speed:5.1f} dosya/s | "
                      f"~{remaining/60:.1f} dk kaldı")

                df_batch   = pd.DataFrame(records)
                write_hdr  = not batch_csv.exists()
                df_batch.to_csv(batch_csv, mode='a', header=write_hdr, index=False)
                records    = []

                save_checkpoint(label_name, done)

    print(f"\n  ✅ {label_name} tamamlandı!\n")
    return batch_csv

def make_graphs(df):
    print("📊 Grafikler oluşturuluyor...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    counts = df['label'].value_counts().sort_index()
    colors = ['#2ecc71', '#e74c3c']
    axes[0].bar(['Benign (0)', 'Malicious (1)'], counts.values,
                color=colors, edgecolor='black')
    axes[0].set_title('Sınıf Dağılımı', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Dosya Sayısı')
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 200, f'{v:,}', ha='center', fontweight='bold')
    axes[1].pie(counts.values, labels=['Benign', 'Malicious'],
                colors=colors, autopct='%1.1f%%', startangle=90)
    axes[1].set_title('Oran', fontsize=14, fontweight='bold')
    plt.suptitle(f'Sınıf Dağılımı | Toplam: {len(df):,}', fontsize=13)
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / 'sinif_dagilimi.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ sinif_dagilimi.png")

    top_f = ['entropy_full', 'obfusc_score', 'dangerous_api_total',
             'api_eval', 'api_atob', 'count_very_long_lines',
             'count_b64_like', 'count_hex_escape', 'api_XMLHttpRequest', 'count_ip_address']
    top_f = [f for f in top_f if f in df.columns]
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    for i, feat in enumerate(top_f[:10]):
        axes[i].boxplot(
            [df[df['label']==0][feat], df[df['label']==1][feat]],
            labels=['Benign', 'Malicious'], patch_artist=True,
            boxprops=dict(facecolor='lightblue'),
            medianprops=dict(color='red', linewidth=2))
        axes[i].set_title(feat, fontsize=9, fontweight='bold')
        axes[i].set_yscale('symlog')
    plt.suptitle('Benign vs Malicious — Feature Dağılımları', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / 'feature_dagilimi.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ feature_dagilimi.png")

    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()['label'].drop('label').abs().sort_values(ascending=False)
    top20 = corr.head(20)
    plt.figure(figsize=(10, 7))
    bars = plt.barh(top20.index[::-1], top20.values[::-1], color='steelblue')
    for bar, val in zip(bars, top20.values[::-1]):
        plt.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                 f'{val:.3f}', va='center', fontsize=8)
    plt.xlabel('|Korelasyon|')
    plt.title('Top 20 Feature — Label Korelasyonu', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / 'korelasyon.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ korelasyon.png")

def main():
    cp = load_checkpoint()
    benign_start    = 0
    malicious_start = 0

    if cp:
        print(f"\n⚡ Checkpoint bulundu! Son durum: {cp['label_name']} — {cp['last_index']} dosya")
        print("   Kaldığı yerden devam ediliyor...\n")
        if cp['label_name'] == 'benign':
            benign_start = cp['last_index']
        elif cp['label_name'] == 'malicious':
            malicious_start = cp['last_index']

    benign_csv = OUTPUT_DIR / 'batch_benign.csv'
    if malicious_start > 0 and benign_csv.exists():
        print("⏩ Benign zaten tamamlanmış, malicious'a geçiliyor...\n")
    else:
        print("\n📋 ADIM 1/4 — Benign dosyaları işleniyor...")
        benign_csv = process_zip(BENIGN_ZIP, label=0,
                                 label_name='benign', start_from=benign_start)

    print("📋 ADIM 2/4 — Malicious dosyaları işleniyor...")
    malicious_csv = process_zip(MALICIOUS_ZIP, label=1,
                                label_name='malicious', start_from=malicious_start)

    print("📋 ADIM 3/4 — Birleştiriliyor ve temizleniyor...")
    df = pd.concat([pd.read_csv(benign_csv), pd.read_csv(malicious_csv)], ignore_index=True)

    if '_error' in df.columns:
        df = df.drop(columns=['_error'])
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)

    df.to_csv(OUTPUT_CSV, index=False)
    benign_csv.unlink(missing_ok=True)
    malicious_csv.unlink(missing_ok=True)
    clear_checkpoint()

    mb = OUTPUT_CSV.stat().st_size / 1024 / 1024
    print(f"  ✅ features.csv → {len(df):,} satır, {len(df.columns)} sütun, {mb:.1f} MB")

    print("\n📋 ADIM 4/4 — Grafikler oluşturuluyor...")
    make_graphs(df)

    print(f"""
{'='*60}
  ✅ HER ŞEY TAMAMLANDI!
{'='*60}
  Çıktı klasörü: {OUTPUT_DIR}

  ├── features.csv            ({mb:.1f} MB)
  └── graphs/
      ├── sinif_dagilimi.png
      ├── feature_dagilimi.png
      └── korelasyon.png

  Sonraki adım: feature_selection.py
{'='*60}
""")

if __name__ == '__main__':
    main()