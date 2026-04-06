import argparse
import os
import gc
import sys
import warnings
import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
from scipy.stats import kendalltau

warnings.filterwarnings("ignore")


KODE_MAP = {
    1473:1472, 9401:9501, 9413:9502, 9414:9503, 9415:9504, 9404:9604,
    9411:9608, 9410:9605, 9412:9601, 9433:9607, 9434:9602, 9435:9606,
    9436:9603, 9402:9702, 9417:9708, 9416:9707, 9418:9704, 9431:9705,
    9432:9706, 9430:9703, 9429:9701, 9107:9202, 9106:9203, 9108:9201,
    9109:9205, 9110:9204, 9171:9271
}

PREDICTORS = [
    'h_hhcount', 'h_nmale', 'h_nfemale',
    'h_ngrad_sd', 'h_ngrad_smp', 'h_ngrad_sma', 'h_ngrad_d', 'h_ngrad_s', 'h_ngrad_s2', 'h_notgrad',
    'h_stat1', 'h_stat2', 'h_stat3', 'h_stat4', 'h_stat5',
    'h_house1', 'h_house2', 'h_house3', 'h_house4',
    'h_floor1', 'h_floor2', 'h_floor3', 'h_floor4',
    'h_wall1', 'h_wall2', 'h_wall3', 'h_wall4',
    'h_roof1', 'h_roof2', 'h_roof3', 'h_roof4',
    'h_dwater1', 'h_dwater2', 'h_dwater3', 'h_dwater4', 'h_dwater5',
    'h_lighting1', 'h_lighting2', 'h_lighting3',
    'h_epower1', 'h_epower2', 'h_epower3',
    'h_asset_lpg5kg', 'h_asset_fridge', 'h_asset_ac',
    'h_asset_computer', 'h_asset_jewelry', 'h_asset_motorcycle',
    'h_asset_car', 'h_asset_land', 'h_lnluaslantai',
    'h_nfamily', 'h_notmarried', 'h_divorced', 'h_widowed',
    'h_toilet1', 'h_toilet2', 'h_toilet3', 'h_toilet4', 'h_toilet5', 'h_toilet6',
    'h_neverschool', 'h_stillschool', 'h_notschool',
    'h_nage04', 'h_nage519', 'h_nage2064', 'h_nage65up',
    'h_asset_internet'
]

FIXED_PARAMS = {
    'objective': 'reg:squarederror',
    'colsample_bytree': 0.3,
    'learning_rate': 0.01,
    'max_depth': 5,
    'n_estimators': 700,
    'subsample': 0.7,
    'reg_lambda': 0.01,
    'random_state': 123,
    'tree_method': 'hist'
}


def compute_decile_ranks(predictions):
    predictions = np.array(predictions)
    sorted_indices = np.argsort(predictions)
    ranks = np.zeros(len(predictions), dtype=np.int32)
    num_samples = len(predictions)
    decile_size = num_samples // 10
    for i in range(10):
        start_idx = i * decile_size
        end_idx = (i + 1) * decile_size if i < 9 else num_samples
        ranks[sorted_indices[start_idx:end_idx]] = i
    return ranks


def compute_group_logic(df, decile_col="decile", suffix=""):
    df[f'group20{suffix}'] = np.where(df[decile_col] <= 1, 0, 1)
    df[f'group40{suffix}'] = np.where(df[decile_col] <= 3, 0, 1)
    df[f'group60{suffix}'] = np.where(df[decile_col] <= 5, 0, 1)
    return df


def inclusion_exclusion(df, pred_group, ref_group):
    cm = confusion_matrix(df[pred_group], df[ref_group], labels=[0, 1])
    tp, fp, fn = cm[0,0], cm[0,1], cm[1,0]
    ie = fp / (tp + fp) if (tp + fp) > 0 else 0
    ee = fn / (tp + fn) if (tp + fn) > 0 else 0
    return ie, ee


def metrics_evaluation(df, pred_col="pred_accum"):
    df_copy = df.copy()
    df_copy['decile_pred'] = compute_decile_ranks(df_copy[pred_col])
    df_copy = compute_group_logic(df_copy, decile_col='decile_pred', suffix='_pred')
    df_copy = compute_group_logic(df_copy, decile_col='decile', suffix='')

    tau_dcl = kendalltau(df_copy['decile_pred'], df_copy['decile']).correlation
    tau20   = kendalltau(df_copy['group20_pred'], df_copy['group20']).correlation
    tau40   = kendalltau(df_copy['group40_pred'], df_copy['group40']).correlation
    tau60   = kendalltau(df_copy['group60_pred'], df_copy['group60']).correlation

    ie20, ee20 = inclusion_exclusion(df_copy, 'group20_pred', 'group20')
    ie40, ee40 = inclusion_exclusion(df_copy, 'group40_pred', 'group40')
    ie60, ee60 = inclusion_exclusion(df_copy, 'group60_pred', 'group60')

    acc20 = accuracy_score(df_copy['group20'], df_copy['group20_pred'])
    acc40 = accuracy_score(df_copy['group40'], df_copy['group40_pred'])
    acc60 = accuracy_score(df_copy['group60'], df_copy['group60_pred'])
    rmse  = np.sqrt(mean_squared_error(df_copy['pcexp'], df_copy[pred_col]))

    return [tau_dcl, tau20, tau40, tau60, ie20, ee20, ie40, ee40, ie60, ee60, acc20, acc40, acc60, rmse]


def load_data(data_files, list_prov, year_test, random_state, outdir):
    print("\n[INFO] Detect provinsi...")

    ref_year = min(data_files.keys())
    ref_path = data_files[ref_year]

    sample = pl.read_csv(ref_path, n_rows=1000)
    p_col = "PROP" if "PROP" in sample.columns else "R101"

    if len(list_prov) == 0:
        list_prov = (
            pl.read_csv(ref_path, columns=[p_col])
            .select(pl.col(p_col).unique())
            .to_series()
            .to_list()
        )

    print(f"[INFO] Total provinsi: {len(list_prov)}")

    results = []

    for prov in list_prov:
        print(f"\n=== PROV {prov} ===")

        temp_list = []

        for year, path in data_files.items():
            if not os.path.exists(path):
                continue

            print(f"[LOAD] {year} - prov {prov}")

            df_pl = pl.scan_csv(path)
            schema = df_pl.collect_schema()
            p_col = "PROP" if "PROP" in schema else "R101"
            k_col = "KAB" if "KAB" in schema else "R102"

            df = (
                df_pl
                .filter(pl.col(p_col) == prov)
                .collect()
                .to_pandas()
            )

            if df.empty:
                continue

            df["year"] = year
            df["kode_kab_new"] = (
                pd.to_numeric(df[p_col]).astype(int) * 100 +
                pd.to_numeric(df[k_col]).astype(int)
            ).replace(KODE_MAP)

            for c in PREDICTORS:
                if c not in df.columns:
                    df[c] = 0.0
                else:
                    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

            temp_list.append(df)

        if len(temp_list) == 0:
            continue

        full_df = pd.concat(temp_list, ignore_index=True)
        full_df = compute_group_logic(full_df, "decile")

        needed = PREDICTORS + ["pcexp", "lpcexp", "decile", "kode_kab_new", "group20", "group40", "group60", "year"]
        full_df = full_df[needed].dropna()

        train_df = full_df[full_df["year"] != year_test]
        test_df  = full_df[full_df["year"] == year_test]

        if len(train_df) == 0 or len(test_df) == 0:
            continue

        kab_list = np.sort(train_df["kode_kab_new"].unique())

        for kab in kab_list:
            tr = train_df[train_df["kode_kab_new"] == kab].copy()
            ts = test_df[test_df["kode_kab_new"] == kab].copy()
            if len(tr) < 50 or len(ts) == 0:
                continue

            print(f"Kab: {kab} | Train: {len(tr)} | Test: {len(ts)}")

            kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
            ts["pred_accum"] = 0.0

            for tr_idx, val_idx in kf.split(tr):
                X_tr, y_tr   = tr.iloc[tr_idx][PREDICTORS], tr.iloc[tr_idx]["lpcexp"]
                X_val, y_val = tr.iloc[val_idx][PREDICTORS], tr.iloc[val_idx]["lpcexp"]

                model = xgb.XGBRegressor(**FIXED_PARAMS, early_stopping_rounds=50)
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

                ts["pred_accum"] += np.exp(model.predict(ts[PREDICTORS])) / 5.0

            m_row = metrics_evaluation(ts, "pred_accum")
            results.append([prov, kab] + m_row)

        del full_df, train_df, test_df, temp_list
        gc.collect()

    cols = ["prov", "Kode_Kab_New", "TauDCL", "Tau20", "Tau40", "Tau60",
            "IE20", "EE20", "IE40", "EE40", "IE60", "EE60",
            "Acc20", "Acc40", "Acc60", "RMSE_Rupiah"]
    report_df = pd.DataFrame(results, columns=cols)
    avg_row = report_df.mean(numeric_only=True).to_frame().T
    avg_row["Kode_Kab_New"] = "AVERAGE"
    final_report = pd.concat([report_df, avg_row], ignore_index=True)

    print("\n=== SUMMARY REPORT ===")
    print(final_report.round(4).to_string(index=False))

    out_path = os.path.join(outdir, f"report_modelling_{year_test}.csv")
    final_report.to_csv(out_path, index=False)
    print(f"\n[INFO] Report tersimpan di: {out_path}")

    return final_report


def main():
    parser = argparse.ArgumentParser(
        description='Modelling XGBoost Susenas PMT',
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument('--data', type=str, nargs='+', required=True, metavar='TAHUN:FILE',
                        help='Data per tahun. Contoh: --data 2022:susenas_clean_2022.csv 2023:susenas_clean_2023.csv')
    parser.add_argument('--year_test', type=int, required=True, help='Tahun yang dipakai sebagai data testing. Contoh: --year_test 2025')
    parser.add_argument('--provinsi', type=int, nargs='*', default=[], help='Filter kode provinsi. Kosongkan untuk semua provinsi.')
    parser.add_argument('--random_state', type=int, default=123, help='Random state (default: 123)')
    parser.add_argument('--output', type=str, default='./output/', help='Folder output report (default: ./output/)')

    args = parser.parse_args()

    data_files = {}
    for item in args.data:
        if ':' not in item:
            print(f"[ERROR] Format salah: {item}. Gunakan format TAHUN:FILE")
            sys.exit(1)
        tahun_str, path = item.split(':', 1)
        try:
            tahun = int(tahun_str)
        except ValueError:
            print(f"[ERROR] Tahun tidak valid: {tahun_str}")
            sys.exit(1)
        if not os.path.exists(path):
            print(f"[WARN] File tidak ditemukan, skip: {path}")
            continue
        data_files[tahun] = path

    if not data_files:
        print("[ERROR] Tidak ada file data yang valid.")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    FIXED_PARAMS['random_state'] = args.random_state

    load_data(
        data_files=data_files,
        list_prov=args.provinsi,
        year_test=args.year_test,
        random_state=args.random_state,
        outdir=args.output
    )

    print("\nSemua proses selesai.")


if __name__ == '__main__':
    main()
