import os
import joblib
import numpy as np
import pandas as pd

FILE_INPUT  = "dtsen_clean_lite_2026.csv"
PATH_MODEL  = "models_final_full_mapbaru/"
PATH_OUTPUT = "hasil_prediksi_mapbaru/"
FINAL_OUTPUT = f"{PATH_OUTPUT}predicted_kabupaten_gianyar_mapbaru.csv"

PREDICTORS = [
    'h_hhcount','h_nmale','h_nfemale','h_ngrad_sd','h_ngrad_smp','h_ngrad_sma',
    'h_ngrad_d','h_ngrad_s','h_ngrad_s2','h_notgrad',
    'h_stat1','h_stat2','h_stat3','h_stat4','h_stat5',
    'h_house1','h_house2','h_house3','h_house4',
    'h_floor1','h_floor2','h_floor3','h_floor4',
    'h_wall1','h_wall2','h_wall3','h_wall4',
    'h_roof1','h_roof2','h_roof3','h_roof4',
    'h_dwater1','h_dwater2','h_dwater3','h_dwater4','h_dwater5',
    'h_lighting1','h_lighting2','h_lighting3',
    'h_epower1','h_epower2','h_epower3',
    'h_asset_lpg5kg','h_asset_fridge','h_asset_ac','h_asset_computer',
    'h_asset_jewelry','h_asset_motorcycle','h_asset_car','h_asset_land',
    'h_lnluaslantai','h_nfamily','h_notmarried','h_divorced','h_widowed',
    'h_toilet1','h_toilet2','h_toilet3','h_toilet4','h_toilet5','h_toilet6',
    'h_neverschool','h_stillschool','h_notschool',
    'h_nage04','h_nage519','h_nage2064','h_nage65up',
    'h_asset_internet'
]


def compute_ranks(df):
    df = df.sort_values('lnpcexp_pred').reset_index(drop=True)

    q_dec  = df['lnpcexp_pred'].quantile([i/10  for i in range(1, 10)]).values
    q_pct  = df['lnpcexp_pred'].quantile([i/100 for i in range(1, 100)]).values

    def rank(x, thresholds):
        return next((i+1 for i, t in enumerate(thresholds) if x <= t), len(thresholds)+1)

    df['decile_pred']     = df['lnpcexp_pred'].apply(lambda x: rank(x, q_dec))
    df['percentile_pred'] = df['lnpcexp_pred'].apply(lambda x: rank(x, q_pct))
    df['group20_pred']    = (df['decile_pred'] > 2).astype(int)
    df['group40_pred']    = (df['decile_pred'] > 4).astype(int)
    df['group60_pred']    = (df['decile_pred'] > 6).astype(int)
    return df


def main():
    os.makedirs(PATH_OUTPUT, exist_ok=True)

    print("[INFO] Loading data...")
    df = pd.read_csv(FILE_INPUT)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df.columns = df.columns.str.lower().str.strip()

    if 'kode_kab' not in df.columns:
        raise ValueError("[ERROR] kolom 'kode_kab' tidak ditemukan")

    df['kode_kab'] = pd.to_numeric(df['kode_kab'], errors='coerce').fillna(0).astype(int)

    for col in PREDICTORS:
        df[col] = pd.to_numeric(df.get(col, 0), errors='coerce').fillna(0)

    unique_kab = sorted(df['kode_kab'].unique())
    print(f"[INFO] Total kabupaten: {len(unique_kab)}")

    results = []
    for kab in unique_kab:
        model_path = f"{PATH_MODEL}xgboost_pmt_full_{kab}.pkl"
        if not os.path.exists(model_path):
            print(f"[SKIP] model {kab} tidak ada")
            continue

        print(f"[PROCESS] kab {kab}")
        df_kab = df[df['kode_kab'] == kab].copy()
        model  = joblib.load(model_path)

        df_kab['lnpcexp_pred'] = model.predict(df_kab[PREDICTORS])
        df_kab['pcexp_pred']   = np.expm1(df_kab['lnpcexp_pred'])
        df_kab = compute_ranks(df_kab)
        results.append(df_kab)

    final_df = pd.concat(results, ignore_index=True)
    final_df.to_csv(FINAL_OUTPUT, index=False)

    print(f"\n=== SELESAI === | Output: {FINAL_OUTPUT} | Total: {len(final_df):,} baris")


if __name__ == '__main__':
    main()
