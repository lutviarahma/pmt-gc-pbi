import pandas as pd
import numpy as np
import os
from functools import reduce

PATH_METERAN = "meteran_listrik_202604080756.csv"
PATH_AK = "ak_nested_202604071642_wo_nik.csv"
PATH_ROOT = "root_table_202604071643_wo_nik.csv"

OUTPUT_STAGE_1 = "2026_dtsen_merged_imputed_gianyar.csv"
OUTPUT_FINAL = "dtsen_clean_lite_2026_gianyar.csv"

# Mapping for Feature Engineering
EDU_LIST = ['h_ngrad_sd', 'h_ngrad_smp', 'h_ngrad_sma', 'h_ngrad_d', 'h_ngrad_s', 'h_ngrad_s2', 'h_notgrad']
SCHOOL_LIST = ['h_neverschool', 'h_stillschool', 'h_notschool']
SEC_LIST = ['h_stat1', 'h_stat2', 'h_stat3', 'h_stat4', 'h_stat5']

CATEGORIES_MAP = {
    'house': ['h_house1', 'h_house2', 'h_house3', 'h_house4'],
    'floor': ['h_floor1', 'h_floor2', 'h_floor3', 'h_floor4'],
    'wall': ['h_wall1', 'h_wall2', 'h_wall3', 'h_wall4'],
    'roof': ['h_roof1', 'h_roof2', 'h_roof3', 'h_roof4'],
    'dwater': ['h_dwater1', 'h_dwater2', 'h_dwater3', 'h_dwater4', 'h_dwater5'],
    'epower': ['h_epower1', 'h_epower2', 'h_epower3'],
    'lighting': ['h_lighting1', 'h_lighting2', 'h_lighting3'],
    'toilet_type': ['h_toilet1', 'h_toilet2', 'h_toilet3', 'h_toilet4', 'h_toilet5', 'h_toilet6']
}

def get_wide(df, col, categories=None):
    if col not in df.columns or df.empty:
        return pd.DataFrame(columns=['ASSIGNMENT_ID'])
    temp_df = df[['ASSIGNMENT_ID', col]].copy()
    temp_df = temp_df[~temp_df[col].isin(['unknown', 'None', 'nan', None, 'UNKNOWN'])]
    counts = temp_df.groupby(['ASSIGNMENT_ID', col], observed=False).size().reset_index(name='count')
    wide = counts.pivot(index='ASSIGNMENT_ID', columns=col, values='count').fillna(0)
    if categories:
        for c in categories:
            if c not in wide.columns: wide[c] = 0
        wide = wide[categories]
    return wide.reset_index()

# =================================================================
# STAGE 1: CLEANING, FILTERING & INITIAL MERGE
# =================================================================
def stage_1_cleaning():
    print("\n[STEP 1] Memulai Integrasi Root, AK, dan Meteran...")

    df_meteran = pd.read_csv(PATH_METERAN, sep=None, engine='python', dtype=str)
    df_ak = pd.read_csv(PATH_AK, sep=None, engine='python', dtype=str)
    df_root_raw = pd.read_csv(PATH_ROOT, sep=None, engine='python', dtype=str)

    for df in [df_meteran, df_ak, df_root_raw]:
        df.columns = df.columns.str.strip().str.lower()
        if 'assignment_id' in df.columns:
            df['assignment_id'] = df['assignment_id'].str.strip()

    # Filter Root (Lantai)
    df_root = df_root_raw.copy()
    if 'jenis_lantai_value' in df_root.columns:
        df_root['jenis_lantai_value'] = df_root['jenis_lantai_value'].replace(['nan', 'None', '', 'NULL'], pd.NA)
        df_root = df_root.dropna(subset=['jenis_lantai_value'])

    # Filter AK (Status Keberadaan 2, 6, 7 dibuang)
    if 'ak_keberadaan_value' in df_ak.columns:
        status_drop = ['2', '6', '7']
        df_ak = df_ak[~df_ak['ak_keberadaan_value'].isin(status_drop)].copy()

    # Agregasi Meteran
    if 'daya_terpasang_value' in df_meteran.columns:
        df_meteran['daya_terpasang_value'] = df_meteran['daya_terpasang_value'].str.strip()
        daya_numeric = pd.to_numeric(df_meteran['daya_terpasang_value'], errors='coerce')
        meteran_agg = df_meteran.assign(daya=daya_numeric).groupby('assignment_id').agg(
            jumlah_meteran=('daya', 'count'),
            daya_maks=('daya', 'max')
        ).reset_index()
        n_multi = (meteran_agg['jumlah_meteran'] > 1).sum()
        print(f"[INFO] Ruta dengan > 1 meteran: {n_multi} ({n_multi/len(meteran_agg)*100:.1f}%)")
    else:
        meteran_agg = pd.DataFrame(columns=['assignment_id', 'daya_maks'])

    # Merging
    ruta_lengkap = df_root.merge(meteran_agg, on='assignment_id', how='left')
    df_merged = ruta_lengkap.merge(df_ak, on='assignment_id', how='inner')

    # Rekonsiliasi: cek ID yang hilang setelah inner merge
    id_root = set(df_root['assignment_id'])
    id_merged = set(df_merged['assignment_id'])
    lost_ids = id_root - id_merged
    print(f"[REKON] Assignment ID di Root tapi tidak ada di AK (hilang setelah inner merge): {len(lost_ids)}")
    if lost_ids:
        pd.Series(list(lost_ids)).to_csv("lost_ids_stage1_gianyar.csv", index=False, header=['assignment_id'])
        print(f"[REKON] Lost IDs disimpan ke lost_ids_stage1_gianyar.csv")

    df_merged.to_csv(OUTPUT_STAGE_1, index=False)
    print(f"[INFO] Stage 1 selesai. Total baris merged: {len(df_merged)}")
    return OUTPUT_STAGE_1

# =================================================================
# STAGE 2: IMPUTATION (NO CHUNKING)
# =================================================================
def stage_2_imputation(merged_file):
    print("\n[STEP 2] Memulai Imputasi...")

    df = pd.read_csv(merged_file, dtype=str)
    df.columns = df.columns.str.lower().str.strip()

    n_before = len(df)

    # Konversi Numerik
    df['ak_umur'] = pd.to_numeric(df['ak_umur'], errors='coerce')
    if 'ak_sekolah_value' in df.columns:
        df['ak_sekolah_value'] = pd.to_numeric(df['ak_sekolah_value'], errors='coerce')

        # Imputasi Balita (Umur 0-4 dianggap belum sekolah)
        mask_balita = df['ak_umur'].between(0, 4, inclusive='both')
        df.loc[mask_balita, 'ak_sekolah_value'] = 0

        # Drop invalid age/school (9 biasanya kode "tidak tahu" atau error)
        df = df[
            df['ak_umur'].notna() &
            df['ak_sekolah_value'].notna() &
            (df['ak_sekolah_value'] != 9)
        ]

    print(f"[INFO] Baris di-drop saat imputasi: {n_before - len(df)} | Sisa: {len(df)}")

    df.to_csv(OUTPUT_STAGE_1, index=False)
    print(f"[INFO] Stage 2 selesai: {OUTPUT_STAGE_1}")

# =================================================================
# STAGE 3: FEATURE ENGINEERING (WIDE FORMAT)
# =================================================================
def stage_3_feature_engineering():
    print("\n[STEP 3] Memulai Feature Engineering & Aggregation...")

    m = {
        'prov': 'level_1_full_code', 'kab': 'level_2_full_code',
        'sch': 'ak_sekolah_value', 'ijz': 'ak_ijazah_value',
        'kawin': 'ak_status_kawin_value', 'sex': 'ak_jk_value',
        'age': 'ak_umur', 'stat': 'ak_status_kerja_value',
        'house': 'status_kepemilikan_rumah_value', 'floor': 'jenis_lantai_value',
        'wall': 'jenis_dinding_value', 'roof': 'jenis_atap_value',
        'water': 'sumber_air_minum_utama_value', 'elec': 'daya_maks',
        'light': 'sumber_penerangan_utama_value', 'toiletA': 'fasilitas_bab_value',
        'luas': 'luas_lantai', 'smart': 'pengeluaran_pulsa'
    }

    df = pd.read_csv(OUTPUT_STAGE_1, dtype=str)
    df.columns = df.columns.str.lower()

    if 'assignment_id' in df.columns:
        df = df.rename(columns={'assignment_id': 'ASSIGNMENT_ID'})

    def get_real_col(key):
        target = m[key]
        for c in df.columns:
            if c == target or c.startswith(target + "_"):
                return c
        return None

    real_m = {k: get_real_col(k) for k in m.keys()}

    # Numeric conversion
    for col in df.columns:
        if col != 'ASSIGNMENT_ID':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # --- INDIVIDUAL LEVEL FEATURES ---
    if real_m['sch']:
        df['school'] = np.select([df[real_m['sch']]==0, df[real_m['sch']]==1, df[real_m['sch']]==2],
                                 ['h_neverschool', 'h_stillschool', 'h_notschool'], default=None)
    if real_m['ijz']:
        ijz = df[real_m['ijz']]
        df['ijazah'] = np.select([ijz==1, ijz==2, ijz==3, ijz==4, ijz==5, ijz==6],
                                 ['h_ngrad_sd', 'h_ngrad_smp', 'h_ngrad_sma', 'h_ngrad_d', 'h_ngrad_s', 'h_ngrad_s2'], default='h_notgrad')
    if real_m['kawin']:
        df['marriage'] = df[real_m['kawin']].map({1:'h_notmarried', 2:'h_married', 3:'h_divorced', 4:'h_widowed'})
    if real_m['sex']:
        df['gender'] = df[real_m['sex']].map({1:'h_nmale', 2:'h_nfemale'})
    if real_m['age']:
        age = df[real_m['age']]
        df['age_cat'] = np.select([age<=4, age<=19, age<=64], ['h_nage04', 'h_nage519', 'h_nage2064'], default='h_nage65up')
    if real_m['stat']:
        df['work_status'] = df[real_m['stat']].map({1:'h_stat1', 2:'h_stat2', 3:'h_stat2', 4:'h_stat3', 5:'h_stat4', 6:'h_stat5'})

    # Aggregating to Household Level
    indiv_base = df.groupby('ASSIGNMENT_ID').agg(h_hhcount=('ASSIGNMENT_ID','size')).reset_index()
    indiv_dfs = [indiv_base]
    for col_name, cats in [('ijazah', EDU_LIST), ('school', SCHOOL_LIST), ('marriage', None), ('gender', None), ('age_cat', None), ('work_status', SEC_LIST)]:
        if col_name in df.columns:
            indiv_dfs.append(get_wide(df, col_name, categories=cats))

    hh_final = reduce(lambda l, r: pd.merge(l, r, on='ASSIGNMENT_ID', how='left'), indiv_dfs).fillna(0)

    # --- HOUSEHOLD LEVEL FEATURES (Assets & Housing) ---
    rt = df.drop_duplicates(subset=['ASSIGNMENT_ID']).copy()
    if real_m['luas']:
        rt['h_luaslantai'] = rt[real_m['luas']].fillna(0)
        rt['h_lnluaslantai'] = np.log1p(rt['h_luaslantai'].replace(0, np.nan)).fillna(0)

    asset_map = {'ac': 'ac', 'lemaries_kulkas': 'fridge', 'gas_5kg': 'lpg5kg', 'komputer_laptop': 'computer', 'emas_perhiasan': 'jewelry', 'sepeda_motor': 'motorcycle', 'mobil': 'car', 'jumlah_lahan_lain': 'land'}
    asset_cols = []
    for key, name in asset_map.items():
        col_name = f'h_asset_{name}'
        if key in rt.columns:
            rt[col_name] = (pd.to_numeric(rt[key], errors='coerce').fillna(0) > 0).astype(int)
            asset_cols.append(col_name)

    if real_m['smart']:
        rt['h_asset_internet'] = np.where(rt[real_m['smart']].fillna(0) > 10000, 1, 0)
        asset_cols.append('h_asset_internet')

    def s_num(col_key):
        col = real_m[col_key]
        return rt[col].fillna(0) if col else pd.Series([0]*len(rt))

    rt['house'] = np.select([s_num('house')==1, s_num('house').isin([3,5]), s_num('house')==2, s_num('house')==4], ['h_house1','h_house2','h_house3','h_house4'], default=None)
    rt['floor'] = np.select([s_num('floor')<=3, s_num('floor')==4, s_num('floor').isin([5,6]), s_num('floor')>=7], ['h_floor1','h_floor2','h_floor3','h_floor4'], default=None)
    rt['wall'] = np.select([s_num('wall')==1, s_num('wall').isin([2,3]), s_num('wall').isin([4,6]), s_num('wall')==7], ['h_wall1','h_wall2','h_wall3','h_wall4'], default=None)
    rt['roof'] = np.select([s_num('roof')==1, s_num('roof')==2, s_num('roof').isin([3,4,5,6]), s_num('roof').isin([7,8])], ['h_roof1','h_roof2','h_roof3','h_roof4'], default=None)
    rt['dwater'] = np.select([s_num('water')==1, s_num('water').isin([2,3]), s_num('water').isin([4,5,7]), s_num('water').isin([6,8]), s_num('water')>=9], ['h_dwater1','h_dwater2','h_dwater3','h_dwater4','h_dwater5'], default=None)
    rt['epower'] = np.select([s_num('elec')==1, s_num('elec')==2, s_num('elec').isin([3,4,5])], ['h_epower1','h_epower2','h_epower3'], default=None)
    rt['lighting'] = np.select([s_num('light')<=2, s_num('light')==3, s_num('light')==4], ['h_lighting1','h_lighting2','h_lighting3'], default=None)
    rt['toilet_type'] = s_num('toiletA').map({1:'h_toilet1', 2:'h_toilet2', 3:'h_toilet3', 4:'h_toilet4', 5:'h_toilet5'}).fillna('h_toilet6')

    id_cols = ['ASSIGNMENT_ID', real_m['prov'], real_m['kab'], 'h_luaslantai', 'h_lnluaslantai', 'house', 'floor', 'wall', 'roof', 'dwater', 'epower', 'lighting', 'toilet_type']
    final = pd.merge(hh_final, rt[id_cols + asset_cols], on='ASSIGNMENT_ID', how='left').fillna(0)

    # One-Hot Encoding for Housing
    for col, cats in CATEGORIES_MAP.items():
        if col in final.columns:
            final[col] = pd.Categorical(final[col], categories=cats)
    final = pd.get_dummies(final, columns=list(CATEGORIES_MAP.keys()), prefix='', prefix_sep='', dtype=int)

    # Normalize Education by Household Count
    for c in EDU_LIST:
        if c in final.columns:
            final[c] = final[c] / final['h_hhcount'].replace(0, 1)

    # Final touch
    final['h_nfamily'] = 1
    final['kode_prov'] = final[real_m['prov']].astype(int)
    final['kode_kab'] = final[real_m['kab']].astype(int)

    final.to_csv(OUTPUT_FINAL, index=False)
    print("\n" + "="*60)
    print("REKONSILIASI GLOBAL")
    print("="*60)
    print(f"Total Rumah Tangga (output final)  : {len(final)}")
    print(f"Total Kolom                        : {len(final.columns)}")
    print(f"Missing values di final            : {final.isnull().sum().sum()}")

    # Cek one-hot konsistensi (tiap kategori harus sum = 1 per baris)
    for cat, cols in CATEGORIES_MAP.items():
        existing = [c for c in cols if c in final.columns]
        if existing:
            row_sum = final[existing].sum(axis=1)
            n_zero = (row_sum == 0).sum()
            n_multi = (row_sum > 1).sum()
            if n_zero > 0 or n_multi > 0:
                print(f"[WARN] {cat}: {n_zero} baris tanpa kategori, {n_multi} baris multi-kategori")
            else:
                print(f"[OK]   {cat}: semua baris terisi tepat 1 kategori")

    # Distribusi asset
    print("\n[INFO] % Kepemilikan Aset:")
    for col in asset_cols:
        if col in final.columns:
            pct = final[col].mean() * 100
            print(f"  {col}: {pct:.1f}%")

    print("="*60)
    print(f"\n[DONE] File Berhasil Dibuat: {OUTPUT_FINAL}")
    print(f"Total Baris (Rumah Tangga): {len(final)}")

if __name__ == "__main__":
    try:
        merged_file = stage_1_cleaning()
        stage_2_imputation(merged_file)
        stage_3_feature_engineering()
    except Exception as e:
        import traceback
        print(f"\n[ERROR] Terjadi kesalahan: {e}")
        traceback.print_exc()
