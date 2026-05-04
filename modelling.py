import argparse
import os
import gc
import sys
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error
from scipy.stats import kendalltau

# Pemekaran wilayah: kode lama (2022) -> kode baru (2023+)
KODE_MAP = {
    1472: 1473,
    9401: 9501, 9413: 9502, 9414: 9503, 9415: 9504,
    9404: 9604, 9411: 9608, 9410: 9605, 9412: 9601,
    9433: 9607, 9434: 9602, 9435: 9606, 9436: 9603,
    9402: 9702, 9417: 9708, 9416: 9707, 9418: 9704,
    9431: 9705, 9432: 9706, 9430: 9703, 9429: 9701,
    9107: 9202, 9106: 9203, 9108: 9201,
    9109: 9205, 9110: 9204, 9171: 9271,
}

# Ekspansi kode provinsi (untuk filter): kode lama -> set kode baru
PROV_EXPAND = {
    91: [91, 92], 92: [91, 92],
    94: [95, 96, 97],
    95: [95, 96, 97], 96: [95, 96, 97], 97: [95, 96, 97],
}

PREDICTORS = [
    'h_hhcount', 'h_nmale', 'h_nfemale',
    'h_ngrad_sd', 'h_ngrad_smp', 'h_ngrad_sma', 'h_ngrad_d',
    'h_ngrad_s', 'h_ngrad_s2', 'h_notgrad',
    'h_stat1', 'h_stat2', 'h_stat3', 'h_stat4', 'h_stat5',
    'h_house1', 'h_house2', 'h_house3', 'h_house4',
    'h_floor1', 'h_floor2', 'h_floor3', 'h_floor4',
    'h_wall1', 'h_wall2', 'h_wall3', 'h_wall4',
    'h_roof1', 'h_roof2', 'h_roof3', 'h_roof4',
    'h_dwater1', 'h_dwater2', 'h_dwater3', 'h_dwater4', 'h_dwater5',
    'h_lighting1', 'h_lighting2', 'h_lighting3',
    'h_epower1', 'h_epower2', 'h_epower3',
    'h_asset_lpg5kg', 'h_asset_fridge', 'h_asset_ac', 'h_asset_computer',
    'h_asset_jewelry', 'h_asset_motorcycle', 'h_asset_car', 'h_asset_land',
    'h_lnluaslantai', 'h_nfamily', 'h_notmarried', 'h_divorced', 'h_widowed',
    'h_toilet1', 'h_toilet2', 'h_toilet3', 'h_toilet4', 'h_toilet5', 'h_toilet6',
    'h_neverschool', 'h_stillschool', 'h_notschool',
    'h_nage04', 'h_nage519', 'h_nage2064', 'h_nage65up',
    'h_asset_internet',
]

XGB_PARAMS = {
    'objective': 'reg:squarederror',
    'colsample_bytree': 0.3,
    'learning_rate': 0.01,
    'max_depth': 5,
    'n_estimators': 700,
    'subsample': 0.7,
    'reg_lambda': 0.01,
    'random_state': 123,
    'tree_method': 'hist',
}


def compute_decile_ranks(values):
    n = len(values)
    if n < 10:
        return np.zeros(n, dtype=np.int32)
    arr = np.asarray(values)
    sorted_idx = np.argsort(arr)
    ranks = np.zeros(n, dtype=np.int32)
    d_size = n // 10
    for i in range(10):
        start = i * d_size
        end = (i + 1) * d_size if i < 9 else n
        ranks[sorted_idx[start:end]] = i
    return ranks


def compute_group_logic(df, decile_col='decile', suffix=''):
    df[f'group20{suffix}'] = np.where(df[decile_col] <= 1, 0, 1)
    df[f'group40{suffix}'] = np.where(df[decile_col] <= 3, 0, 1)
    df[f'group60{suffix}'] = np.where(df[decile_col] <= 5, 0, 1)
    return df


def inclusion_exclusion(df, pred_group, ref_group):
    cm = confusion_matrix(df[ref_group], df[pred_group], labels=[0, 1])
    tp, fp, fn = cm[0, 0], cm[0, 1], cm[1, 0]
    ie = fp / (tp + fp) if (tp + fp) > 0 else 0
    ee = fn / (tp + fn) if (tp + fn) > 0 else 0
    return ie, ee


def metrics_evaluation(df, pred_col='pred_accum'):
    df = df.copy()
    df['decile_pred'] = compute_decile_ranks(df[pred_col].values)
    df = compute_group_logic(df, decile_col='decile_pred', suffix='_pred')

    tau_dcl = kendalltau(df['decile_pred'], df['decile']).correlation
    tau20 = kendalltau(df['group20_pred'], df['group20']).correlation
    tau40 = kendalltau(df['group40_pred'], df['group40']).correlation
    tau60 = kendalltau(df['group60_pred'], df['group60']).correlation

    ie20, ee20 = inclusion_exclusion(df, 'group20_pred', 'group20')
    ie40, ee40 = inclusion_exclusion(df, 'group40_pred', 'group40')
    ie60, ee60 = inclusion_exclusion(df, 'group60_pred', 'group60')

    acc20 = accuracy_score(df['group20'], df['group20_pred'])
    acc40 = accuracy_score(df['group40'], df['group40_pred'])
    acc60 = accuracy_score(df['group60'], df['group60_pred'])
    rmse = np.sqrt(mean_squared_error(df['pcexp'], df[pred_col]))

    metrics = [tau_dcl, tau20, tau40, tau60,
               ie20, ee20, ie40, ee40, ie60, ee60,
               acc20, acc40, acc60, rmse]
    return metrics, df


def load_year(path, year):
    df = pd.read_csv(path)
    p_col = next((c for c in ['PROP', 'R101'] if c in df.columns), 'R101')
    k_col = next((c for c in ['KAB', 'R102'] if c in df.columns), 'R102')
    if p_col not in df.columns or k_col not in df.columns:
        print(f"[WARN] {path}: kolom prov/kab tidak ditemukan ({p_col}, {k_col}).")
    df['year'] = year
    df['kode_kab_new'] = (df[p_col].astype(int) * 100
                          + df[k_col].astype(int)).replace(KODE_MAP)
    df['kode_prov_new'] = df['kode_kab_new'] // 100
    return df


def parse_data_args(items):
    out = {}
    for it in items:
        if ':' not in it:
            print(f"[WARN] Format --data salah (skip): {it}. Pakai YEAR:FILE")
            continue
        y, p = it.split(':', 1)
        out[int(y)] = p
    return out


def main():
    parser = argparse.ArgumentParser(
        description='Modelling XGBoost PMT Susenas (per-kabupaten)',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--data', type=str, nargs='+', required=True,
                        metavar='YEAR:FILE',
                        help='Multi-tahun. Contoh: --data 2022:susenas_clean_2022.csv 2023:susenas_clean_2023.csv')
    parser.add_argument('--year_test', type=int, required=True,
                        help='Tahun yang dipakai sebagai test set.')
    parser.add_argument('--provinsi', type=int, nargs='*', default=[],
                        help='Filter kode provinsi. Kosongkan = semua.')
    parser.add_argument('--output', type=str, default='./output')
    parser.add_argument('--no_save_models', action='store_true',
                        help='Tidak menyimpan model .pkl per kab.')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    model_dir = os.path.join(args.output, 'models')
    os.makedirs(model_dir, exist_ok=True)

    data_files = parse_data_args(args.data)
    if args.year_test not in data_files:
        print(f"[ERROR] year_test {args.year_test} tidak ada di --data")
        sys.exit(1)

    print("[INFO] Loading data...")
    pool = []
    for y, p in sorted(data_files.items()):
        if not os.path.exists(p):
            print(f"  [WARN] {p} tidak ditemukan, skip.")
            continue
        df = load_year(p, y)
        pool.append(df)
        print(f"  [OK] {y}: {len(df):,} baris ({p})")
    if not pool:
        print("[ERROR] Tidak ada data yang berhasil dimuat.")
        sys.exit(1)

    full = pd.concat(pool, ignore_index=True)
    del pool
    gc.collect()
    print(f"[INFO] Total: {len(full):,} baris\n")

    if args.provinsi:
        prov_set = set()
        for p in args.provinsi:
            prov_set.add(p)
            prov_set.update(PROV_EXPAND.get(p, []))
        prov_to_run = sorted(prov_set)
        full = full[full['kode_prov_new'].isin(prov_set)]
        print(f"[INFO] Filter provinsi (after expand): {prov_to_run}")
    else:
        prov_to_run = sorted(full['kode_prov_new'].unique())
        print(f"[INFO] Mode penuh: {len(prov_to_run)} provinsi")

    master_csv = os.path.join(args.output, f'master_testing_{args.year_test}.csv')
    if os.path.exists(master_csv):
        os.remove(master_csv)

    summary = []
    save_models = not args.no_save_models

    for prov in prov_to_run:
        df_prov = full[full['kode_prov_new'] == prov].copy()
        if df_prov.empty:
            continue
        df_prov = compute_group_logic(df_prov, 'decile')
        kabs = sorted(df_prov['kode_kab_new'].unique())
        print(f"\n[RUN] Provinsi {prov} ({len(kabs)} kab)")
        prov_test_buf = []

        for kab in kabs:
            df_kab = df_prov[df_prov['kode_kab_new'] == kab].reset_index(drop=True)
            df_train = df_kab.copy()
            df_test = df_kab[df_kab['year'] == args.year_test].copy()

            if len(df_train) < 100 or df_test.empty:
                continue

            for m in PREDICTORS:
                if m not in df_train.columns:
                    df_train[m] = 0
                if m not in df_test.columns:
                    df_test[m] = 0

            model = xgb.XGBRegressor(**XGB_PARAMS)
            model.fit(df_train[PREDICTORS], df_train['lpcexp'])
            if save_models:
                joblib.dump(model, os.path.join(model_dir, f'xgb_{kab}.pkl'))

            df_test['pred_log'] = model.predict(df_test[PREDICTORS])
            df_test['pred_accum'] = np.expm1(df_test['pred_log'])
            df_test['y_hat_rupiah'] = df_test['pred_accum']

            mrow, df_test_eval = metrics_evaluation(df_test, 'pred_accum')
            summary.append([prov, kab] + mrow)
            prov_test_buf.append(df_test_eval)
            print(f"   Kab {kab} | train={len(df_train):,} test={len(df_test):,} "
                  f"| RMSE={mrow[-1]:.0f} TauDCL={mrow[0]:.3f}")

        if prov_test_buf:
            pd.concat(prov_test_buf).to_csv(
                master_csv, mode='a', index=False,
                header=not os.path.exists(master_csv),
            )
        gc.collect()

    cols = ['Prov', 'Kab', 'TauDCL', 'Tau20', 'Tau40', 'Tau60',
            'IE20', 'EE20', 'IE40', 'EE40', 'IE60', 'EE60',
            'Acc20', 'Acc40', 'Acc60', 'RMSE']
    report = pd.DataFrame(summary, columns=cols)

    if not report.empty:
        avg = report.mean(numeric_only=True).to_frame().T
        avg['Prov'] = 'AVERAGE'
        avg['Kab'] = 'AVERAGE'
        report_full = pd.concat([report, avg], ignore_index=True)
    else:
        report_full = report

    report_path = os.path.join(args.output, f'report_modelling_{args.year_test}.csv')
    report_full.to_csv(report_path, index=False)

    print(f"\n[SELESAI]")
    print(f"  Models  : {model_dir}")
    print(f"  Master  : {master_csv}")
    print(f"  Report  : {report_path}")


if __name__ == '__main__':
    main()
