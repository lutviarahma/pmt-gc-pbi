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

csv.field_size_limit(sys.maxsize)


COLUMN_MAPPING = {
    2025: dict(
        c_prov='PROP', c_kab='KAB', c_sch='R611', c_ijz='R614',
        c_kawin='R404', c_sex='R405', c_age='R407',
        c_sec='R705', c_stat='R706', c_jam='R707',
        c_house='R1602', c_floor='R1608', c_wall='R1607', c_roof='R1606',
        c_water='R1610A', c_elec='R1616B1', c_light='R1616', c_fuel='R1617',
        c_toiletA='R1609A', c_luas='R1604', asset_prefix='R1801',
        c_smart='R802', c_family='R1601', c_internet='R808'
    ),
    2024: dict(
        c_prov='R101', c_kab='R102', c_sch='R610', c_ijz='R614',
        c_kawin='R404', c_sex='R405', c_age='R407',
        c_sec='R706', c_stat='R707', c_jam='R708',
        c_house='R1802', c_floor='R1808', c_wall='R1807', c_roof='R1806A',
        c_water='R1810A', c_elec='R1816B1', c_light='R1816', c_fuel='R1817',
        c_toiletA='R1809A', c_luas='R1804', asset_prefix='R2001',
        c_smart='R802', c_family='R1801', c_internet='R808'
    ),
    2023: dict(
        c_prov='R101', c_kab='R102', c_sch='R610', c_ijz='R614',
        c_kawin='R404', c_sex='R405', c_age='R407',
        c_sec='R706', c_stat='R707', c_jam='R708',
        c_house='R1802', c_floor='R1808', c_wall='R1807', c_roof='R1806',
        c_water='R1810A', c_elec='R1816B1', c_light='R1816', c_fuel='R1817',
        c_toiletA='R1809A', c_luas='R1804', asset_prefix='R2001',
        c_smart='R802', c_family='R1801', c_internet='R808'
    ),
    2022: dict(
        c_prov='R101', c_kab='R102', c_sch='R610', c_ijz='R614',
        c_kawin='R404', c_sex='R405', c_age='R407',
        c_sec='R705', c_stat='R706', c_jam='R707',
        c_house='R1802', c_floor='R1808', c_wall='R1807', c_roof='R1806',
        c_water='R1810A', c_elec='R1816B1', c_light='R1816', c_fuel='R1817',
        c_toiletA='R1809A', c_luas='R1804', asset_prefix='R2001',
        c_smart='R802', c_family='R1801', c_internet='R808'
    ),
}


def detect_separator(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        first_line = f.readline()
    if '|' in first_line:
        return '|'
    elif ';' in first_line and first_line.count(';') > first_line.count(','):
        return ';'
    return ','


def normalize_columns(df):
    df.columns = [re.sub(r'[^a-zA-Z0-9_]', '', str(c)).strip().upper() for c in df.columns]
    df = df.rename(columns={col: col[:-2] for col in df.columns if col.endswith('_X')})
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
            print(f"[ERROR] pyarrow tidak tersedia. Install: pip install pyarrow")
            return None
        df = pd.read_parquet(path)
    else:
        sep = detect_separator(path)
        df = pd.read_csv(path, sep=sep, engine='python', dtype=str, on_bad_lines='skip', low_memory=False)
    df = normalize_columns(df)
    return df


def load_file_chunked(path, chunksize=50000):
    if path.endswith('.parquet'):
        if not PARQUET_AVAILABLE:
            print(f"[ERROR] pyarrow tidak tersedia. Install: pip install pyarrow")
            return
        p_file = pq.ParquetFile(path)
        for batch in p_file.iter_batches(batch_size=chunksize):
            yield batch.to_pandas()
    else:
        sep = detect_separator(path)
        reader = pd.read_csv(path, sep=sep, engine='python', dtype=str, chunksize=chunksize, on_bad_lines='skip', low_memory=False)
        for chunk in reader:
            yield chunk


def compute_decile_ranks(vals):
    if len(vals) < 10:
        return np.zeros(len(vals), dtype=int)
    sorted_idx = np.argsort(vals)
    ranks = np.empty(len(vals), dtype=int)
    n = len(vals)
    d_size = n // 10
    for i in range(10):
        start = i * d_size
        end = (i + 1) * d_size if i < 9 else n
        ranks[sorted_idx[start:end]] = i
    return ranks


def get_wide(df, col, categories=None):
    if col not in df.columns or df.empty:
        return pd.DataFrame(columns=['URUT'])
    temp_series = df[col].astype(object).fillna('unknown')
    counts = (
        df.assign(temp_col=temp_series)
        .groupby(['URUT', 'temp_col'], observed=False)
        .size()
        .reset_index(name='count')
    )
    wide = counts.pivot(index='URUT', columns='temp_col', values='count').fillna(0)
    if categories is not None:
        for c in categories:
            if c not in wide.columns:
                wide[c] = 0
    if 'unknown' in wide.columns:
        wide = wide.drop(columns=['unknown'])
    return wide.reset_index()


def join_kor_kp(kor_path, kp_path, output_folder="."):
    print(f"\n[JOIN] KOR + KP")

    df_kor = load_file(kor_path)
    df_kp  = load_file(kp_path)

    if df_kor is None or df_kp is None:
        print("[ERROR] File KOR atau KP tidak ditemukan.")
        return None

    df_kor = normalize_urut(df_kor)
    df_kp  = normalize_urut(df_kp)

    cols_to_use = df_kp.columns.difference(df_kor.columns).tolist() + ['URUT']
    df_raw_ruta = df_kor.merge(df_kp[cols_to_use], on='URUT', how='left')

    out_path = os.path.join(output_folder, "raw_ruta.csv")
    df_raw_ruta.to_csv(out_path, index=False)
    print(f" -> Selesai! File tersimpan di: {out_path}")

    del df_kor, df_kp, df_raw_ruta; gc.collect()
    return out_path


def merge_ruta_individu(ruta_path, individu_path, output_folder=".", year=None):
    print(f"\n[MERGE] Ruta + Individu{f' tahun {year}' if year else ''}")

    if not individu_path or not os.path.exists(individu_path):
        print(f"[ERROR] File individu tidak ditemukan: {individu_path}")
        return None

    out_path = os.path.join(output_folder, f"{str(year)[-2:]}_susenas_merged_{year}.csv")

    if os.path.exists(out_path):
        os.remove(out_path)

    df_ruta = None
    if ruta_path and os.path.exists(ruta_path):
        print(f" -> Loading ruta: {ruta_path}")
        df_ruta = load_file(ruta_path)
        df_ruta = normalize_urut(df_ruta)
        df_ruta = df_ruta.drop_duplicates(subset=['URUT'])
        print(f" -> Data ruta siap ({len(df_ruta)} baris unik).")
        gc.collect()

    first_chunk = True
    print(f" -> Merging individu...")

    for chunk in load_file_chunked(individu_path):
        chunk = normalize_columns(chunk)
        chunk = normalize_urut(chunk)

        if df_ruta is not None:
            cols_to_use = list(df_ruta.columns.difference(chunk.columns)) + ['URUT']
            merged = chunk.merge(df_ruta[cols_to_use], on='URUT', how='left')
        else:
            merged = chunk

        merged.to_csv(out_path, index=False, mode='w' if first_chunk else 'a', header=first_chunk)
        first_chunk = False
        del chunk, merged; gc.collect()

    print(f" -> BERHASIL! File tersimpan: {out_path}")
    return out_path


def preprocess(merged_path, year, col, output_folder=".", provinsi_filter=None, chunk_size=100000):
    provinsi_filter = provinsi_filter or []

    if not os.path.exists(merged_path):
        print(f"[ERROR] File tidak ditemukan: {merged_path}")
        return None

    print(f"\n[PREPO] Susenas {year} | File: {merged_path}")

    c_prov       = col['c_prov']
    c_kab        = col['c_kab']
    c_sch        = col['c_sch']
    c_ijz        = col['c_ijz']
    c_kawin      = col['c_kawin']
    c_sex        = col['c_sex']
    c_age        = col['c_age']
    c_sec        = col['c_sec']
    c_stat       = col['c_stat']
    c_jam        = col['c_jam']
    c_house      = col['c_house']
    c_floor      = col['c_floor']
    c_wall       = col['c_wall']
    c_roof       = col['c_roof']
    c_water      = col['c_water']
    c_elec       = col['c_elec']
    c_light      = col['c_light']
    c_fuel       = col['c_fuel']
    c_toiletA    = col['c_toiletA']
    c_luas       = col['c_luas']
    asset_prefix = col['asset_prefix']
    c_smart      = col['c_smart']
    c_family     = col['c_family']
    c_internet   = col['c_internet']

    EDU_LIST    = ['h_ngrad_sd', 'h_ngrad_smp', 'h_ngrad_sma', 'h_ngrad_d', 'h_ngrad_s', 'h_ngrad_s2', 'h_notgrad']
    SCHOOL_LIST = ['h_neverschool', 'h_stillschool', 'h_notschool']
    SEC_LIST = ['h_stat1', 'h_stat2', 'h_stat3', 'h_stat4', 'h_stat5']
    asset_map   = {
        'A': 'lpg5kg', 'B': 'fridge', 'C': 'ac', 'D': 'wheater', 'E': 'phone',
        'F': 'computer', 'G': 'jewelry', 'H': 'motorcycle', 'I': 'boat',
        'J': 'motorboat', 'K': 'car', 'L': 'tv', 'M': 'land'
    }

    all_chunks_final = []

    for chunk in load_file_chunked(merged_path, chunksize=chunk_size):
        chunk.columns = [str(c).replace('"', '').replace("'", "").strip().upper() for c in chunk.columns]
        chunk = chunk.rename(columns={c: c[:-2] for c in chunk.columns if c.endswith('_X')})
        chunk = chunk.apply(lambda x: x.str.strip() if x.dtype == 'object' else x).replace('', np.nan)

        def clean_special(col_name, fill_value):
            if col_name in chunk.columns:
                chunk[col_name] = chunk[col_name].astype(str).str.replace(' ', str(fill_value))
                chunk[col_name] = pd.to_numeric(chunk[col_name], errors='coerce').fillna(fill_value).astype(int)

        clean_special(c_sec, 1)
        clean_special(c_elec, 1)
        clean_special(c_stat, 1)

        if 'URUT' not in chunk.columns:
            continue

        target_candidates = ['KAPITA', 'PCEXP', 'KAPITA_X']
        found_target = next((c for c in target_candidates if c in chunk.columns), None)
        chunk['pcexp'] = pd.to_numeric(chunk[found_target], errors='coerce').fillna(0) if found_target else 0

        must_be_numeric = [c_sec, c_stat, c_jam, c_age, c_ijz, c_light, c_elec, c_prov, c_kab, c_toiletA, c_luas]
        for c in must_be_numeric:
            if c in chunk.columns:
                chunk[c] = pd.to_numeric(chunk[c], errors='coerce').fillna(0)

        if provinsi_filter:
            chunk = chunk[chunk[c_prov].astype(float).isin(provinsi_filter)]
        if chunk.empty:
            continue

        indiv_dfs = []

        # Partisipasi sekolah
        if c_sch in chunk.columns:
            chunk[c_sch] = pd.to_numeric(chunk[c_sch], errors='coerce')
            chunk.loc[chunk[c_sch] == 1, 'school'] = 'h_neverschool'
            chunk.loc[chunk[c_sch] == 2, 'school'] = 'h_stillschool'
            chunk.loc[chunk[c_sch] == 3, 'school'] = 'h_notschool'
            chunk['school'] = pd.Categorical(chunk['school'], categories=SCHOOL_LIST)
            school_df = (
                chunk[['URUT', c_sch, 'school']]
                .groupby(['URUT', 'school'])
                .count()
                .sort_values(['URUT', 'school'])
                .reset_index()
            )
            school_wide = school_df.pivot(index='URUT', columns='school', values=c_sch).reset_index()
            indiv_dfs.append(school_wide)

        # Ijazah
        if c_ijz in chunk.columns:
            chunk[c_ijz] = pd.to_numeric(chunk[c_ijz], errors='coerce').fillna(25).astype(int)
            chunk['ijazah'] = np.select(
                [
                    chunk[c_ijz] <= 5,
                    (chunk[c_ijz] > 5)  & (chunk[c_ijz] <= 10),
                    (chunk[c_ijz] > 10) & (chunk[c_ijz] <= 17),
                    chunk[c_ijz].isin([18, 19, 20]),
                    chunk[c_ijz].isin([21, 22]),
                    chunk[c_ijz].isin([23, 24]),
                    chunk[c_ijz] == 25
                ],
                ['h_ngrad_sd', 'h_ngrad_smp', 'h_ngrad_sma', 'h_ngrad_d', 'h_ngrad_s', 'h_ngrad_s2', 'h_notgrad'],
                default='h_notgrad'
            )

        # Status pernikahan
        if c_kawin in chunk.columns:
            chunk['marriage'] = chunk[c_kawin].astype(float).map({1: 'h_notmarried', 2: 'h_married', 3: 'h_divorced', 4: 'h_widowed'})

        # Jenis kelamin
        if c_sex in chunk.columns:
            chunk['gender'] = chunk[c_sex].astype(float).map({1: 'h_nmale', 2: 'h_nfemale'})

        # Binning umur
        if c_age in chunk.columns:
            chunk['age_cat'] = np.select(
                [chunk[c_age] <= 4, chunk[c_age] <= 19, chunk[c_age] <= 64, chunk[c_age] > 64],
                ['h_nage04', 'h_nage519', 'h_nage2064', 'h_nage65up'],
                default='h_nage_unknown'
            )

        # Status pekerjaan
        if c_stat in chunk.columns:
            status_map = {1: 'h_stat1', 2: 'h_stat2', 3: 'h_stat2', 4: 'h_stat3', 5: 'h_stat4', 6: 'h_stat5', 7: 'h_stat6'}
            chunk['work_status'] = pd.to_numeric(chunk[c_stat], errors='coerce').map(status_map)

        # Jumlah ART
        agg_dict = {'h_hhcount': ('URUT', 'size')}
        if c_jam in chunk.columns:
            agg_dict['h_avg_workhours'] = (c_jam, 'mean')
        indiv_base = chunk.groupby('URUT').agg(**agg_dict).reset_index()
        indiv_dfs.append(indiv_base)

        wide_configs = [
            ('ijazah', EDU_LIST), ('school', SCHOOL_LIST),
            ('marriage', None), ('gender', None), ('age_cat', None), ('work_status', None), ('school_cat', None), ('work_status', SEC_LIST)
        ]
        for col_name, cats in wide_configs:
            if col_name in chunk.columns:
                indiv_dfs.append(get_wide(chunk, col_name, categories=cats))

        indiv_final = reduce(lambda l, r: pd.merge(l, r, on='URUT', how='left'), indiv_dfs).fillna(0)

        # Data Rumah Tangga
        rt = chunk.drop_duplicates(subset=['URUT']).copy()
        rt['pcexp'] = pd.to_numeric(rt['pcexp'], errors='coerce').fillna(0)
        rt['lpcexp'] = np.log1p(rt['pcexp'])
        rt['h_nfamily'] = pd.to_numeric(rt.get(c_family, 1), errors='coerce').fillna(1)
        rt['h_luaslantai'] = pd.to_numeric(rt.get(c_luas, 0), errors='coerce').fillna(0)
        rt['h_lnluaslantai'] = np.log(rt['h_luaslantai'].replace(0, np.nan)).fillna(0)

        for suffix, name in asset_map.items():
            col_key = f"{asset_prefix}{suffix}"
            rt[f'h_asset_{name}'] = np.where(pd.to_numeric(rt.get(col_key, 0), errors='coerce') == 1, 1, 0)

        # Internet & smart device digabung
        available_tech = [c for c in [c_internet, c_smart] if c in chunk.columns]
        if available_tech:
            for c in available_tech:
                chunk[f'{c}_bin'] = np.where(pd.to_numeric(chunk[c], errors='coerce') == 1, 1, 0)
            bin_cols = [f'{c}_bin' for c in available_tech]
            agg_tech = chunk.groupby('URUT')[bin_cols].max().max(axis=1).reset_index(name='h_asset_internet')
            rt = rt.merge(agg_tech, on='URUT', how='left')
        rt['h_asset_internet'] = rt['h_asset_internet'].fillna(0).astype(int) if 'h_asset_internet' in rt.columns else 0

        def s_num(c): return pd.to_numeric(rt.get(c, 0), errors='coerce').fillna(0)

        rt['house']    = np.select([s_num(c_house)==1, s_num(c_house).isin([3,5]), s_num(c_house)==2, s_num(c_house)==4], ['h_house1','h_house2','h_house3','h_house4'], default='h_house1')
        rt['floor']    = np.select([s_num(c_floor)<=3, s_num(c_floor)==4, s_num(c_floor).isin([5,6]), s_num(c_floor)>=7], ['h_floor1','h_floor2','h_floor3','h_floor4'], default='h_floor1')
        rt['wall']     = np.select([s_num(c_wall)==1, s_num(c_wall).isin([2,3]), s_num(c_wall).isin([4,6]), s_num(c_wall)==7], ['h_wall1','h_wall2','h_wall3','h_wall4'], default='h_wall1')
        rt['roof']     = np.select([s_num(c_roof)==1, s_num(c_roof)==2, s_num(c_roof).isin([3,4,5,6]), s_num(c_roof).isin([7,8])], ['h_roof1','h_roof2','h_roof3','h_roof4'], default='h_roof1')
        rt['dwater']   = np.select([s_num(c_water)==1, s_num(c_water).isin([2,3]), s_num(c_water).isin([4,5,7]), s_num(c_water).isin([6,8])], ['h_dwater1','h_dwater2','h_dwater3','h_dwater4'], default='h_dwater5')
        rt['epower']   = np.select([s_num(c_elec)==1, s_num(c_elec)==2], ['h_epower1','h_epower2'], default='h_epower3')
        rt['lighting'] = np.select([s_num(c_light)<=2, s_num(c_light)==3, s_num(c_light)==4], ['h_lighting1','h_lighting2','h_lighting3'], default='h_lighting1')

        fuel = pd.to_numeric(rt[c_fuel], errors='coerce')
        rt['cookingfuel'] = np.nan
        rt.loc[fuel == 1, 'cookingfuel'] = 'h_cookingfuel1'
        rt.loc[fuel == 3, 'cookingfuel'] = 'h_cookingfuel2'
        rt.loc[fuel.isin([2, 4, 6]), 'cookingfuel'] = 'h_cookingfuel3'
        rt.loc[fuel.isin([7, 8, 9, 10, 11]), 'cookingfuel'] = 'h_cookingfuel4'
        rt.loc[fuel == 0, 'cookingfuel'] = 'h_cookingfuel5'
        rt['cookingfuel'] = pd.Categorical(rt['cookingfuel'], categories=['h_cookingfuel1','h_cookingfuel2','h_cookingfuel3','h_cookingfuel4','h_cookingfuel5'])

        if c_toiletA in rt.columns:
            toilet = pd.to_numeric(rt[c_toiletA], errors='coerce')
            rt['toilet_type'] = toilet.map({1:'h_toilet1', 2:'h_toilet2', 3:'h_toilet3', 4:'h_toilet4', 5:'h_toilet5', 6:'h_toilet6'}).fillna('h_toilet6')

        asset_cols = [f'h_asset_{n}' for n in asset_map.values()] + ['h_asset_internet']
        id_cols = ['URUT', c_prov, c_kab, 'pcexp', 'lpcexp', 'h_luaslantai', 'h_lnluaslantai',
                   'house', 'floor', 'wall', 'roof', 'dwater', 'epower', 'lighting',
                   'cookingfuel', 'toilet_type', 'h_nfamily']

        chunk_final = pd.merge(indiv_final, rt[id_cols + asset_cols], on='URUT', how='outer')

        for col in list(chunk_final.columns):
            if col.endswith('_x'):
                base = col[:-2]
                col_y = base + '_y'

                if col_y in chunk_final.columns:
                    chunk_final[base] = chunk_final[col].combine_first(chunk_final[col_y])
                    chunk_final.drop([col, col_y], axis=1, inplace=True)
                else:
                    chunk_final.rename(columns={col: base}, inplace=True)

        chunk_final = chunk_final[[c for c in chunk_final.columns if not c.endswith('_y')]]
        
        numeric_cols = chunk_final.select_dtypes(include=np.number).columns
        chunk_final[numeric_cols] = chunk_final[numeric_cols].fillna(0)

        all_chunks_final.append(chunk_final)
        del chunk, rt, indiv_final; gc.collect()

    final = pd.concat(all_chunks_final, ignore_index=True)
    del all_chunks_final; gc.collect()

    num_cols = final.select_dtypes(include=[np.number]).columns.tolist()
    if 'URUT' in num_cols:
        num_cols.remove('URUT')

    rt_level_vars = ['pcexp', 'lpcexp', 'h_luaslantai', 'h_lnluaslantai', c_prov, c_kab, 'h_nfamily']
    agg_rules = {
        c: ('max' if c in rt_level_vars else 'sum' if c in num_cols else 'first')
        for c in final.columns if c != 'URUT'
    }
    final = final.groupby('URUT').agg(agg_rules).reset_index()

    for c in EDU_LIST:
        if c in final.columns:
            final[c] = final[c] / final['h_hhcount']

    categories = {
        'house':       ['h_house1','h_house2','h_house3','h_house4'],
        'floor':       ['h_floor1','h_floor2','h_floor3','h_floor4'],
        'wall':        ['h_wall1','h_wall2','h_wall3','h_wall4'],
        'roof':        ['h_roof1','h_roof2','h_roof3','h_roof4'],
        'dwater':      ['h_dwater1','h_dwater2','h_dwater3','h_dwater4','h_dwater5'],
        'epower':      ['h_epower1','h_epower2','h_epower3'],
        'lighting':    ['h_lighting1','h_lighting2','h_lighting3'],
        'cookingfuel': ['h_cookingfuel1','h_cookingfuel2','h_cookingfuel3','h_cookingfuel4','h_cookingfuel5'],
        'toilet_type': ['h_toilet1','h_toilet2','h_toilet3','h_toilet4','h_toilet5','h_toilet6']
    }

    for col_name, cats in categories.items():
        if col_name in final.columns:
            final[col_name] = pd.Categorical(final[col_name], categories=cats)

    final = pd.get_dummies(final, columns=list(categories.keys()), prefix='', prefix_sep='')

    dummy_cols = [c for sublist in categories.values() for c in sublist]
    for col_name in dummy_cols:
        if col_name in final.columns:
            final[col_name] = final[col_name].astype(int)

    final['kode_kab'] = final[c_prov].astype(int) * 100 + final[c_kab].astype(int)

    final['decile'] = 0
    for kab in final['kode_kab'].unique():
        mask = final['kode_kab'] == kab
        final.loc[mask, 'decile'] = compute_decile_ranks(final.loc[mask, 'pcexp'].values)

    out_path = os.path.join(output_folder, f"susenas_clean_{year}.csv")
    final.to_csv(out_path, index=False)
    print(f"DONE: {out_path} | Baris: {len(final)} | Avg Exp: {final['pcexp'].mean():.0f}")
    del final; gc.collect()
    return out_path


def parse_custom_map(map_args):
    custom = {}
    for item in map_args:
        if '=' not in item:
            print(f"[WARN] Format mapping tidak valid (skip): {item}")
            continue
        k, v = item.split('=', 1)
        custom[k.strip()] = v.strip()
    return custom


def get_column_mapping(year, custom_map=None):
    if custom_map:
        base = COLUMN_MAPPING.get(year, {}).copy()
        base.update(custom_map)
        return base
    if year in COLUMN_MAPPING:
        return COLUMN_MAPPING[year].copy()
    print(f"[ERROR] Tahun {year} tidak ada di mapping. Gunakan --map untuk custom mapping.")
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Preprocessing Susenas PMT',
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument('--tahun', required=True, type=int, help='Tahun data. Contoh: --tahun 2025')
    parser.add_argument('--output', type=str, default='.', help='Folder output (default: direktori saat ini)')
    parser.add_argument('--provinsi', type=int, nargs='*', default=[], help='Filter kode provinsi. Contoh: --provinsi 34 31')
    parser.add_argument('--chunk', type=int, default=100000, help='Chunk size untuk hemat RAM (default: 100000)')
    parser.add_argument('--map', type=str, nargs='*', default=[], metavar='KEY=VALUE',
                        help='Custom mapping kolom untuk tahun baru. Contoh: --map c_prov=PROP c_kab=KAB')

    skenario = parser.add_mutually_exclusive_group(required=True)
    skenario.add_argument('--kor', type=str, metavar='FILE', help='Path file KOR (gunakan bersama --kp)')
    skenario.add_argument('--ruta', type=str, metavar='FILE', help='Path file raw_ruta (gunakan bersama --individu)')
    skenario.add_argument('--merged', type=str, metavar='FILE', help='Path file yang sudah merged (langsung ke preprocessing)')

    parser.add_argument('--kp', type=str, metavar='FILE', help='Path file KP (digunakan bersama --kor)')
    parser.add_argument('--individu', type=str, metavar='FILE', help='Path file individu (digunakan bersama --ruta atau --kor)')

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    custom_map = parse_custom_map(args.map) if args.map else None
    col = get_column_mapping(args.tahun, custom_map)
    if col is None:
        sys.exit(1)

    print(f"\n{'='*50}")
    print(f"TAHUN {args.tahun}")
    print(f"{'='*50}")

    if args.kor:
        if not args.kp:
            print("[ERROR] --kor harus digunakan bersama --kp")
            sys.exit(1)
        raw_ruta_path = join_kor_kp(args.kor, args.kp, output_folder=args.output)
        if not raw_ruta_path:
            sys.exit(1)
        if args.individu:
            merged_path = merge_ruta_individu(
                ruta_path=raw_ruta_path,
                individu_path=args.individu,
                output_folder=args.output,
                year=args.tahun
            )
            if not merged_path:
                sys.exit(1)
        else:
            merged_path = raw_ruta_path
        preprocess(merged_path, args.tahun, col, args.output, args.provinsi, args.chunk)

    elif args.ruta:
        if not args.individu:
            print("[ERROR] --ruta harus digunakan bersama --individu")
            sys.exit(1)
        merged_path = merge_ruta_individu(
            ruta_path=args.ruta,
            individu_path=args.individu,
            output_folder=args.output,
            year=args.tahun
        )
        if not merged_path:
            sys.exit(1)
        preprocess(merged_path, args.tahun, col, args.output, args.provinsi, args.chunk)

    elif args.merged:
        preprocess(args.merged, args.tahun, col, args.output, args.provinsi, args.chunk)

    print("\nSemua proses selesai.")


if __name__ == '__main__':
    main()
