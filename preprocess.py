import argparse
import os
import gc
import re
import sys
import csv
import warnings
import numpy as np
import pandas as pd
from functools import reduce

warnings.filterwarnings("ignore")

try:
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False

# Bypass csv field size limit untuk kolom panjang
_max_int = sys.maxsize
while True:
    try:
        csv.field_size_limit(_max_int)
        break
    except OverflowError:
        _max_int = int(_max_int / 10)


COLUMN_MAPPING = {
    2025: dict(
        c_prov='PROP', c_kab='KAB', c_sch='R611', c_ijz='R614',
        c_kawin='R404', c_sex='R405', c_age='R407',
        c_sec='R705', c_stat='R706', c_jam='R707',
        c_house='R1602', c_floor='R1608', c_wall='R1607', c_roof='R1606',
        c_water='R1610A', c_elec='R1616B1', c_light='R1616', c_fuel='R1617',
        c_toiletA='R1609A', c_luas='R1604', asset_prefix='R1801',
        c_smart='R802', c_family='R1601', c_internet='R808',
    ),
    2024: dict(
        c_prov='R101', c_kab='R102', c_sch='R610', c_ijz='R614',
        c_kawin='R404', c_sex='R405', c_age='R407',
        c_sec='R706', c_stat='R707', c_jam='R708',
        c_house='R1802', c_floor='R1808', c_wall='R1807', c_roof='R1806A',
        c_water='R1810A', c_elec='R1816B1', c_light='R1816', c_fuel='R1817',
        c_toiletA='R1809A', c_luas='R1804', asset_prefix='R2001',
        c_smart='R802', c_family='R1801', c_internet='R808',
    ),
    2023: dict(
        c_prov='R101', c_kab='R102', c_sch='R610', c_ijz='R614',
        c_kawin='R404', c_sex='R405', c_age='R407',
        c_sec='R706', c_stat='R707', c_jam='R708',
        c_house='R1802', c_floor='R1808', c_wall='R1807', c_roof='R1806',
        c_water='R1810A', c_elec='R1816B1', c_light='R1816', c_fuel='R1817',
        c_toiletA='R1809A', c_luas='R1804', asset_prefix='R2001',
        c_smart='R802', c_family='R1801', c_internet='R808',
    ),
    2022: dict(
        c_prov='R101', c_kab='R102', c_sch='R610', c_ijz='R614',
        c_kawin='R404', c_sex='R405', c_age='R407',
        c_sec='R705', c_stat='R706', c_jam='R707',
        c_house='R1802', c_floor='R1808', c_wall='R1807', c_roof='R1806',
        c_water='R1810A', c_elec='R1816B1', c_light='R1816', c_fuel='R1817',
        c_toiletA='R1809A', c_luas='R1804', asset_prefix='R2001',
        c_smart='R802', c_family='R1801', c_internet='R808',
    ),
}


EDU_LIST = ['h_ngrad_sd', 'h_ngrad_smp', 'h_ngrad_sma',
            'h_ngrad_d', 'h_ngrad_s', 'h_ngrad_s2', 'h_notgrad']
SCHOOL_LIST = ['h_neverschool', 'h_stillschool', 'h_notschool']
SEC_LIST = ['h_stat1', 'h_stat2', 'h_stat3', 'h_stat4', 'h_stat5']
ASSET_MAP = {
    'A': 'lpg5kg', 'B': 'fridge', 'C': 'ac', 'D': 'wheater', 'E': 'phone',
    'F': 'computer', 'G': 'jewelry', 'H': 'motorcycle', 'I': 'boat',
    'J': 'motorboat', 'K': 'car', 'L': 'tv', 'M': 'land',
}
CATEGORIES = {
    'house':       ['h_house1', 'h_house2', 'h_house3', 'h_house4'],
    'floor':       ['h_floor1', 'h_floor2', 'h_floor3', 'h_floor4'],
    'wall':        ['h_wall1', 'h_wall2', 'h_wall3', 'h_wall4'],
    'roof':        ['h_roof1', 'h_roof2', 'h_roof3', 'h_roof4'],
    'dwater':      ['h_dwater1', 'h_dwater2', 'h_dwater3', 'h_dwater4', 'h_dwater5'],
    'epower':      ['h_epower1', 'h_epower2', 'h_epower3'],
    'lighting':    ['h_lighting1', 'h_lighting2', 'h_lighting3'],
    'cookingfuel': ['h_cookingfuel1', 'h_cookingfuel2', 'h_cookingfuel3',
                    'h_cookingfuel4', 'h_cookingfuel5'],
    'toilet_type': ['h_toilet1', 'h_toilet2', 'h_toilet3',
                    'h_toilet4', 'h_toilet5', 'h_toilet6'],
}


def detect_separator(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        first_line = f.readline()
    if '|' in first_line:
        return '|'
    if ';' in first_line and first_line.count(';') > first_line.count(','):
        return ';'
    return ','


def normalize_columns(df):
    df.columns = [re.sub(r'[^a-zA-Z0-9_]', '', str(c)).strip().upper() for c in df.columns]
    df = df.rename(columns={c: c[:-2] for c in df.columns if c.endswith('_X')})
    return df


def normalize_urut(df):
    u_col = next((c for c in df.columns if 'URUT' in c or 'IDRT' in c), None)
    if u_col and u_col != 'URUT':
        df.rename(columns={u_col: 'URUT'}, inplace=True)
    if 'URUT' in df.columns:
        df['URUT'] = pd.to_numeric(df['URUT'], errors='coerce').fillna(0).astype(np.int64)
    return df


def load_file(path):
    if not path or not os.path.exists(path):
        return None
    if path.endswith('.parquet'):
        if not PARQUET_AVAILABLE:
            print("[ERROR] pyarrow tidak tersedia. Install: pip install pyarrow")
            return None
        df = pd.read_parquet(path)
    else:
        sep = detect_separator(path)
        df = pd.read_csv(path, sep=sep, dtype=str, on_bad_lines='skip', low_memory=False)
    return normalize_columns(df)


def load_file_chunked(path, chunksize=50000):
    if path.endswith('.parquet'):
        if not PARQUET_AVAILABLE:
            print("[ERROR] pyarrow tidak tersedia. Install: pip install pyarrow")
            return
        p_file = pq.ParquetFile(path)
        for batch in p_file.iter_batches(batch_size=chunksize):
            yield batch.to_pandas()
    else:
        sep = detect_separator(path)
        for chunk in pd.read_csv(path, sep=sep, dtype=str, chunksize=chunksize,
                                 on_bad_lines='skip', low_memory=False):
            yield chunk


def compute_decile_ranks(vals):
    n = len(vals)
    if n < 10:
        return np.zeros(n, dtype=int)
    sorted_idx = np.argsort(vals)
    ranks = np.empty(n, dtype=int)
    d_size = n // 10
    for i in range(10):
        start = i * d_size
        end = (i + 1) * d_size if i < 9 else n
        ranks[sorted_idx[start:end]] = i
    return ranks


def get_wide(df, col, categories=None):
    if col not in df.columns or df.empty:
        return pd.DataFrame(columns=['URUT'])
    temp = df[['URUT', col]].copy()
    temp[col] = temp[col].astype(str).replace(['nan', 'None', '', ' '], 'unknown')
    counts = temp.groupby(['URUT', col], observed=False).size().reset_index(name='count')
    wide = counts.pivot(index='URUT', columns=col, values='count').fillna(0)
    if categories:
        for c in categories:
            if c not in wide.columns:
                wide[c] = 0
        wide = wide[categories]
    if 'unknown' in wide.columns:
        wide = wide.drop(columns=['unknown'])
    return wide.reset_index()


def join_kor_kp(kor_path, kp_path, output_folder=".", year=None):
    print("\n[JOIN] KOR + KP")
    df_kor = load_file(kor_path)
    df_kp = load_file(kp_path)
    if df_kor is None or df_kp is None:
        print("[ERROR] File KOR atau KP tidak ditemukan.")
        return None

    df_kor = normalize_urut(df_kor)
    df_kp = normalize_urut(df_kp)

    cols_to_use = df_kp.columns.difference(df_kor.columns).tolist() + ['URUT']
    df_raw_ruta = df_kor.merge(df_kp[cols_to_use], on='URUT', how='left')

    out_name = f"raw_ruta_{year}.csv" if year else "raw_ruta.csv"
    out_path = os.path.join(output_folder, out_name)
    df_raw_ruta.to_csv(out_path, index=False)

    n_total = len(df_raw_ruta)
    n_filled = df_raw_ruta['URUT'].notna().sum()
    print(f" -> {out_path} | total {n_total:,} | URUT terisi {n_filled:,}")

    del df_kor, df_kp, df_raw_ruta
    gc.collect()
    return out_path


def merge_ruta_individu(ruta_path, individu_path, output_folder=".", year=None):
    print(f"\n[MERGE] Ruta + Individu{f' tahun {year}' if year else ''}")
    if not individu_path or not os.path.exists(individu_path):
        print(f"[ERROR] File individu tidak ditemukan: {individu_path}")
        return None

    out_name = f"{str(year)[-2:]}_susenas_merged_{year}.csv" if year else "susenas_merged.csv"
    out_path = os.path.join(output_folder, out_name)
    if os.path.exists(out_path):
        os.remove(out_path)

    df_ruta = None
    if ruta_path and os.path.exists(ruta_path):
        print(f" -> Loading ruta: {ruta_path}")
        df_ruta = load_file(ruta_path)
        df_ruta = normalize_urut(df_ruta)
        df_ruta = df_ruta.drop_duplicates(subset=['URUT'])
        print(f" -> Ruta siap ({len(df_ruta):,} baris unik).")
        gc.collect()

    first_chunk = True
    tail_buffer = pd.DataFrame()
    print(" -> Merging individu (chunked)...")

    for chunk in load_file_chunked(individu_path):
        chunk = normalize_columns(chunk)
        chunk = normalize_urut(chunk)
        if 'URUT' not in chunk.columns:
            continue

        if not tail_buffer.empty:
            chunk = pd.concat([tail_buffer, chunk], ignore_index=True)

        last_id = chunk['URUT'].iloc[-1]
        mask_tail = chunk['URUT'] == last_id
        tail_buffer = chunk[mask_tail].copy()
        chunk = chunk[~mask_tail]

        if not chunk.empty:
            if df_ruta is not None:
                cols_to_use = list(df_ruta.columns.difference(chunk.columns)) + ['URUT']
                merged = chunk.merge(df_ruta[cols_to_use], on='URUT', how='left')
            else:
                merged = chunk
            merged.to_csv(out_path, index=False,
                          mode='w' if first_chunk else 'a', header=first_chunk)
            first_chunk = False
            del merged
        del chunk
        gc.collect()

    if not tail_buffer.empty:
        if df_ruta is not None:
            cols_to_use = list(df_ruta.columns.difference(tail_buffer.columns)) + ['URUT']
            merged = tail_buffer.merge(df_ruta[cols_to_use], on='URUT', how='left')
        else:
            merged = tail_buffer
        merged.to_csv(out_path, index=False,
                      mode='w' if first_chunk else 'a', header=first_chunk)

    print(f" -> BERHASIL! {out_path}")
    return out_path


def process_block(chunk, col, year):
    c = col

    target_candidates = ['KAPITA', 'PCEXP', 'KAPITA_X']
    found = next((t for t in target_candidates if t in chunk.columns), None)
    chunk['pcexp'] = pd.to_numeric(chunk[found], errors='coerce').fillna(0) if found else 0

    must_be_numeric = [c['c_sec'], c['c_stat'], c['c_jam'], c['c_age'], c['c_ijz'],
                       c['c_light'], c['c_elec'], c['c_prov'], c['c_kab'],
                       c['c_toiletA'], c['c_luas']]
    for cn in must_be_numeric:
        if cn in chunk.columns:
            chunk[cn] = pd.to_numeric(chunk[cn], errors='coerce').fillna(0)

    if c['c_sch'] in chunk.columns:
        sch = pd.to_numeric(chunk[c['c_sch']], errors='coerce')
        chunk['school'] = np.select(
            [sch == 1, sch == 2, sch == 3],
            ['h_neverschool', 'h_stillschool', 'h_notschool'],
            default=None,
        )

    if c['c_ijz'] in chunk.columns:
        ijz = pd.to_numeric(chunk[c['c_ijz']], errors='coerce').fillna(25)
        chunk['ijazah'] = np.select(
            [(ijz >= 1) & (ijz <= 5), (ijz > 5) & (ijz <= 10),
             (ijz > 10) & (ijz <= 17), ijz.isin([18, 19, 20]),
             ijz.isin([21, 22]), ijz.isin([23, 24])],
            ['h_ngrad_sd', 'h_ngrad_smp', 'h_ngrad_sma',
             'h_ngrad_d', 'h_ngrad_s', 'h_ngrad_s2'],
            default='h_notgrad',
        )

    if c['c_kawin'] in chunk.columns:
        chunk['marriage'] = pd.to_numeric(chunk[c['c_kawin']], errors='coerce').map(
            {1: 'h_notmarried', 2: 'h_married', 3: 'h_divorced', 4: 'h_widowed'}
        )
    if c['c_sex'] in chunk.columns:
        chunk['gender'] = pd.to_numeric(chunk[c['c_sex']], errors='coerce').map(
            {1: 'h_nmale', 2: 'h_nfemale'}
        )
    if c['c_age'] in chunk.columns:
        age = chunk[c['c_age']]
        chunk['age_cat'] = np.select(
            [age <= 4, age <= 19, age <= 64, age > 64],
            ['h_nage04', 'h_nage519', 'h_nage2064', 'h_nage65up],
            default=None,
        )
    if c['c_stat'] in chunk.columns:
        chunk['work_status'] = pd.to_numeric(chunk[c['c_stat']], errors='coerce').map(
            {1: 'h_stat1', 2: 'h_stat2', 3: 'h_stat2',
             4: 'h_stat3', 5: 'h_stat4', 6: 'h_stat5'}
        )

    agg_dict = {'h_hhcount': ('URUT', 'size')}
    if c['c_jam'] in chunk.columns:
        agg_dict['h_avg_workhours'] = (c['c_jam'], 'mean')
    indiv_base = chunk.groupby('URUT').agg(**agg_dict).reset_index()

    indiv_dfs = [indiv_base]
    wide_configs = [
        ('ijazah', EDU_LIST), ('school', SCHOOL_LIST),
        ('marriage', None), ('gender', None),
        ('age_cat', None), ('work_status', SEC_LIST),
    ]
    for col_name, cats in wide_configs:
        if col_name in chunk.columns:
            indiv_dfs.append(get_wide(chunk, col_name, categories=cats))
    indiv_final = reduce(lambda l, r: pd.merge(l, r, on='URUT', how='left'),
                         indiv_dfs).fillna(0)

    rt = chunk.drop_duplicates(subset=['URUT']).copy()
    rt['lpcexp'] = np.log1p(rt['pcexp'])
    rt['h_nfamily'] = pd.to_numeric(rt.get(c['c_family'], 1), errors='coerce').fillna(1)
    rt['h_luaslantai'] = pd.to_numeric(rt.get(c['c_luas'], 0), errors='coerce').fillna(0)
    rt['h_lnluaslantai'] = np.log1p(rt['h_luaslantai'].replace(0, np.nan)).fillna(0)

    asset_cols = []
    for suffix, name in ASSET_MAP.items():
        col_key = f"{c['asset_prefix']}{suffix}"
        col_name = f'h_asset_{name}'
        rt[col_name] = np.where(
            pd.to_numeric(rt.get(col_key, 0), errors='coerce') == 1, 1, 0)
        asset_cols.append(col_name)

    tech_cols = [t for t in [c['c_internet'], c['c_smart']] if t in chunk.columns]
    if tech_cols:
        for t in tech_cols:
            chunk[f'{t}_bin'] = np.where(
                pd.to_numeric(chunk[t], errors='coerce') == 1, 1, 0)
        bin_cols = [f'{t}_bin' for t in tech_cols]
        agg_tech = (chunk.groupby('URUT')[bin_cols].max()
                    .max(axis=1).reset_index(name='h_asset_internet'))
        rt = rt.merge(agg_tech, on='URUT', how='left')
        rt['h_asset_internet'] = rt['h_asset_internet'].fillna(0).astype(int)
    else:
        rt['h_asset_internet'] = 0

    def s_num(cn):
        return pd.to_numeric(rt.get(cn, 0), errors='coerce').fillna(0)

    rt['house'] = np.select(
        [s_num(c['c_house']) == 1, s_num(c['c_house']).isin([3, 5]),
         s_num(c['c_house']) == 2, s_num(c['c_house']) == 4],
        ['h_house1', 'h_house2', 'h_house3', 'h_house4'], default=None)
    rt['floor'] = np.select(
        [s_num(c['c_floor']) <= 3, s_num(c['c_floor']) == 4,
         s_num(c['c_floor']).isin([5, 6]), s_num(c['c_floor']) >= 7],
        ['h_floor1', 'h_floor2', 'h_floor3', 'h_floor4'], default=None)
    rt['wall'] = np.select(
        [s_num(c['c_wall']) == 1, s_num(c['c_wall']).isin([2, 3]),
         s_num(c['c_wall']).isin([4, 6]), s_num(c['c_wall']) == 7],
        ['h_wall1', 'h_wall2', 'h_wall3', 'h_wall4'], default=None)
    rt['roof'] = np.select(
        [s_num(c['c_roof']) == 1, s_num(c['c_roof']) == 2,
         s_num(c['c_roof']).isin([3, 4, 5, 6]), s_num(c['c_roof']).isin([7, 8])],
        ['h_roof1', 'h_roof2', 'h_roof3', 'h_roof4'], default=None)
    rt['dwater'] = np.select(
        [s_num(c['c_water']) == 1, s_num(c['c_water']).isin([2, 3]),
         s_num(c['c_water']).isin([4, 5, 7]), s_num(c['c_water']).isin([6, 8]),
         s_num(c['c_water']) >= 9],
        ['h_dwater1', 'h_dwater2', 'h_dwater3', 'h_dwater4', 'h_dwater5'],
        default=None)
    rt['epower'] = np.select(
        [s_num(c['c_elec']) == 1, s_num(c['c_elec']) == 2, s_num(c['c_elec']) >= 3],
        ['h_epower1', 'h_epower2', 'h_epower3'], default=None)
    rt['lighting'] = np.select(
        [s_num(c['c_light']) <= 2, s_num(c['c_light']) == 3, s_num(c['c_light']) == 4],
        ['h_lighting1', 'h_lighting2', 'h_lighting3'], default=None)
    rt['cookingfuel'] = np.select(
        [s_num(c['c_fuel']) == 1, s_num(c['c_fuel']) == 3,
         s_num(c['c_fuel']).isin([2, 4, 6]),
         s_num(c['c_fuel']).isin([7, 8, 9, 10, 11]),
         s_num(c['c_fuel']) == 0],
        ['h_cookingfuel1', 'h_cookingfuel2', 'h_cookingfuel3',
         'h_cookingfuel4', 'h_cookingfuel5'],
        default=None)
    rt['toilet_type'] = s_num(c['c_toiletA']).map(
        {1: 'h_toilet1', 2: 'h_toilet2', 3: 'h_toilet3',
         4: 'h_toilet4', 5: 'h_toilet5'}
    ).fillna('h_toilet6')

    id_cols = ['URUT', c['c_prov'], c['c_kab'], 'pcexp', 'lpcexp',
               'h_luaslantai', 'h_lnluaslantai', 'h_nfamily',
               'house', 'floor', 'wall', 'roof', 'dwater',
               'epower', 'lighting', 'cookingfuel', 'toilet_type']

    return pd.merge(indiv_final,
                    rt[id_cols + asset_cols + ['h_asset_internet']],
                    on='URUT', how='outer').fillna(0)


def preprocess(merged_path, year, col, output_folder=".",
               provinsi_filter=None, chunk_size=100000):
    provinsi_filter = provinsi_filter or []
    if not os.path.exists(merged_path):
        print(f"[ERROR] File tidak ditemukan: {merged_path}")
        return None

    print(f"\n[PREPO] Susenas {year} | File: {merged_path}")

    all_chunks = []
    tail_buffer = pd.DataFrame()

    def _clean_special(df):
        for c_fix in [col['c_sec'], col['c_stat'], col['c_elec']]:
            if c_fix in df.columns:
                df[c_fix] = (df[c_fix].astype(str).str.strip()
                             .replace(['', 'nan', 'None'], '1'))
        return df

    for chunk in load_file_chunked(merged_path, chunksize=chunk_size):
        chunk.columns = [str(x).replace('"', '').replace("'", "").strip().upper()
                         for x in chunk.columns]
        chunk = chunk.rename(columns={x: x[:-2] for x in chunk.columns if x.endswith('_X')})
        chunk = normalize_urut(chunk)
        if 'URUT' not in chunk.columns:
            continue

        if not tail_buffer.empty:
            chunk = pd.concat([tail_buffer, chunk], ignore_index=True)

        last_urut = chunk['URUT'].iloc[-1]
        mask_tail = chunk['URUT'] == last_urut
        tail_buffer = chunk[mask_tail].copy()
        chunk = chunk[~mask_tail]
        if chunk.empty:
            continue

        chunk = _clean_special(chunk)
        if provinsi_filter:
            chunk = chunk[pd.to_numeric(chunk[col['c_prov']],
                                        errors='coerce').isin(provinsi_filter)]
        if chunk.empty:
            continue

        all_chunks.append(process_block(chunk, col, year))
        del chunk
        gc.collect()

    if not tail_buffer.empty:
        tail_buffer = _clean_special(tail_buffer)
        if provinsi_filter:
            tail_buffer = tail_buffer[pd.to_numeric(
                tail_buffer[col['c_prov']], errors='coerce').isin(provinsi_filter)]
        if not tail_buffer.empty:
            all_chunks.append(process_block(tail_buffer, col, year))

    if not all_chunks:
        print("[ERROR] Tidak ada data hasil preprocessing.")
        return None

    final = pd.concat(all_chunks, ignore_index=True)
    del all_chunks
    gc.collect()

    num_cols = final.select_dtypes(include=[np.number]).columns.tolist()
    if 'URUT' in num_cols:
        num_cols.remove('URUT')

    rt_level = ['pcexp', 'lpcexp', 'h_luaslantai', 'h_lnluaslantai',
                col['c_prov'], col['c_kab'], 'h_nfamily']
    for r in rt_level:
        if r in final.columns:
            final[r] = pd.to_numeric(final[r], errors='coerce').fillna(0)

    agg_rules = {}
    for cn in final.columns:
        if cn == 'URUT':
            continue
        if cn in rt_level or cn.startswith('h_asset_'):
            agg_rules[cn] = 'max'
        elif cn in num_cols:
            agg_rules[cn] = 'sum'
        else:
            agg_rules[cn] = 'first'
    final = final.groupby('URUT').agg(agg_rules).reset_index()

    for ec in EDU_LIST:
        if ec in final.columns:
            final[ec] = final[ec] / final['h_hhcount'].replace(0, 1)

    for col_name, cats in CATEGORIES.items():
        if col_name in final.columns:
            final[col_name] = pd.Categorical(final[col_name], categories=cats)
    final = pd.get_dummies(final, columns=list(CATEGORIES.keys()),
                           prefix='', prefix_sep='', dtype=int)

    final['kode_kab'] = (final[col['c_prov']].astype(int) * 100
                         + final[col['c_kab']].astype(int))
    final['decile'] = final.groupby('kode_kab')['pcexp'].transform(
        lambda x: compute_decile_ranks(x.values)
    )

    out_path = os.path.join(output_folder, f"susenas_clean_{year}.csv")
    final.to_csv(out_path, index=False)
    print(f"DONE: {out_path} | Baris: {len(final):,} | Avg Exp: {final['pcexp'].mean():.0f}")
    del final
    gc.collect()
    return out_path


def parse_custom_map(items):
    out = {}
    for it in items:
        if '=' not in it:
            print(f"[WARN] Format mapping salah (skip): {it}")
            continue
        k, v = it.split('=', 1)
        out[k.strip()] = v.strip()
    return out


def get_column_mapping(year, custom_map=None):
    base = COLUMN_MAPPING.get(year, {}).copy()
    if custom_map:
        base.update(custom_map)
    if not base:
        print(f"[ERROR] Tahun {year} tidak ada di mapping & --map kosong.")
        return None
    return base


def main():
    parser = argparse.ArgumentParser(
        description='Preprocessing Susenas PMT (single-year, single-skenario)',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--tahun', required=True, type=int,
                        help='Tahun data. Contoh: --tahun 2025')
    parser.add_argument('--output', type=str, default='.',
                        help='Folder output (default: .)')
    parser.add_argument('--provinsi', type=int, nargs='*', default=[],
                        help='Filter kode provinsi. Contoh: --provinsi 34 31')
    parser.add_argument('--chunk', type=int, default=100000,
                        help='Chunk size (default: 100000)')
    parser.add_argument('--map', type=str, nargs='*', default=[],
                        metavar='KEY=VALUE',
                        help='Custom mapping kolom. Contoh: --map c_prov=PROP c_kab=KAB')

    sk = parser.add_mutually_exclusive_group(required=True)
    sk.add_argument('--kor', type=str, metavar='FILE',
                    help='Path KOR (gunakan bersama --kp)')
    sk.add_argument('--ruta', type=str, metavar='FILE',
                    help='Path raw_ruta (gunakan bersama --individu)')
    sk.add_argument('--merged', type=str, metavar='FILE',
                    help='Path file sudah merged')

    parser.add_argument('--kp', type=str, metavar='FILE',
                        help='Path KP (digunakan bersama --kor)')
    parser.add_argument('--individu', type=str, metavar='FILE',
                        help='Path individu (digunakan bersama --kor atau --ruta)')

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    custom_map = parse_custom_map(args.map) if args.map else None
    col = get_column_mapping(args.tahun, custom_map)
    if col is None:
        sys.exit(1)

    print(f"\n{'=' * 50}\nTAHUN {args.tahun}\n{'=' * 50}")

    if args.kor:
        if not args.kp:
            print("[ERROR] --kor harus digunakan bersama --kp")
            sys.exit(1)
        raw_ruta = join_kor_kp(args.kor, args.kp, args.output, year=args.tahun)
        if not raw_ruta:
            sys.exit(1)
        if args.individu:
            merged = merge_ruta_individu(raw_ruta, args.individu,
                                         args.output, args.tahun)
            if not merged:
                sys.exit(1)
        else:
            merged = raw_ruta
        preprocess(merged, args.tahun, col, args.output,
                   args.provinsi, args.chunk)

    elif args.ruta:
        if not args.individu:
            print("[ERROR] --ruta harus digunakan bersama --individu")
            sys.exit(1)
        merged = merge_ruta_individu(args.ruta, args.individu,
                                     args.output, args.tahun)
        if not merged:
            sys.exit(1)
        preprocess(merged, args.tahun, col, args.output,
                   args.provinsi, args.chunk)

    else:
        preprocess(args.merged, args.tahun, col, args.output,
                   args.provinsi, args.chunk)

    print("\nSemua proses selesai.")


if __name__ == '__main__':
    main()
