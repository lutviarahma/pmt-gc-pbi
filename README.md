# PMT Susenas - Preprocessing & Modelling

Pipeline preprocessing dan modelling untuk data Susenas PMT (disesuaikan dgn kuesioner Sensus Ekonomi 2026).

## File

- `preprocess.py` - pipeline preprocessing (load -> output `susenas_clean_{year}.csv`)
- `modelling.py` - pipeline modelling XGBoost, input dari hasil preprocess
- `run_preprocess.py` - interface terpusat, tinggal lempar data + tahun

## Alur Data

```
Skenario 1 (KOR + KP terpisah):
KOR + KP -> join -> raw_ruta + individu -> merge -> prepo -> susenas_clean_{year}.csv

Skenario 2 (ruta + individu terpisah):
raw_ruta + individu -> merge -> prepo -> susenas_clean_{year}.csv

Skenario 3 (sudah merged):
data_merged -> prepo -> susenas_clean_{year}.csv
```

## run_preprocess.py

User tinggal pilih skenario, masukin file dan tahun.

Skenario 3, data sudah merged (2023, 2024):
```bash
python run_preprocess.py merged \
  --tahun 2023 2024 \
  --file 23_susenas_kor.csv 24_susenas_kor.csv \
  --output ./hasil
```

Skenario 1, KOR + KP terpisah, ada data individu (2025):
```bash
python run_preprocess.py kor \
  --tahun 2025 \
  --kor 25_susenas_individu.csv \
  --kp 25_susenas_clean.csv \
  --individu 25_ind.csv \
  --output ./hasil
```

Skenario 1, KOR + KP terpisah tanpa individu:
```bash
python run_preprocess.py kor \
  --tahun 2025 \
  --kor 25_susenas_individu.csv \
  --kp 25_susenas_clean.csv \
  --output ./hasil
```

Skenario 2, ruta + individu terpisah (2022):
```bash
python run_preprocess.py ruta \
  --tahun 2022 \
  --ruta 22_ruta.csv \
  --individu 22_ind.csv \
  --output ./hasil
```

Beberapa tahun sekaligus (cuma di subcommand `merged`):
```bash
python run_preprocess.py merged \
  --tahun 2023 2024 \
  --file 23_susenas_kor.csv 24_susenas_kor.csv
```

Custom mapping kolom utk tahun baru:
```bash
python run_preprocess.py merged \
  --tahun 2026 \
  --file 26_susenas_merged.csv \
  --map c_prov=PROP c_kab=KAB c_sch=R700 c_ijz=R614
```

## preprocess.py

Bisa juga dijalankan langsung per tahun.

Skenario 3, langsung ke preprocessing:
```bash
python preprocess.py --tahun 2023 --merged 23_susenas_kor.csv
python preprocess.py --tahun 2024 --merged 24_susenas_kor.csv
```

Skenario 1, KOR + KP + individu:
```bash
python preprocess.py --tahun 2025 \
  --kor 25_susenas_individu.csv \
  --kp 25_susenas_clean.csv \
  --individu 25_ind.csv
```

Skenario 2, ruta + individu:
```bash
python preprocess.py --tahun 2022 \
  --ruta 22_ruta.csv \
  --individu 22_ind.parquet
```

Filter provinsi + custom chunk size:
```bash
python preprocess.py --tahun 2025 --merged data.csv --provinsi 34 31 --chunk 50000
```

Custom mapping kolom:
```bash
python preprocess.py --tahun 2026 --merged data.csv \
  --map c_prov=PROP c_kab=KAB c_sch=R700 c_ijz=R614 c_stat=R706
```

Output: `susenas_clean_{year}.csv`

## modelling.py

Input dari hasil `preprocess.py`. Loop per provinsi & per kabupaten otomatis. Train XGBoost pakai semua tahun yg dilempar, test di tahun yg dipilih (`--year_test`).

Semua provinsi:
```bash
python modelling.py \
  --data 2022:susenas_clean_2022.csv 2023:susenas_clean_2023.csv 2024:susenas_clean_2024.csv \
  --year_test 2024 \
  --output ./output
```

Filter provinsi tertentu:
```bash
python modelling.py \
  --data 2022:susenas_clean_2022.csv 2023:susenas_clean_2023.csv 2024:susenas_clean_2024.csv \
  --year_test 2024 \
  --provinsi 34 31 12 \
  --output ./output
```

Output:
- `output/report_modelling_{year_test}.csv` - laporan metrik per kabupaten
- `output/master_testing_{year_test}.csv` - hasil prediksi gabungan
- `output/models/xgb_{kab}.pkl` - model per kabupaten (pakai `--no_save_models` kalau ga mau disimpan)

## Format Input

- CSV & Parquet, auto-detect dari ekstensi
- Separator CSV auto-detect (`,` / `;` / `|`)
- Nama kolom otomatis dinormalisasi (uppercase, karakter non-alphanumeric dihapus)

## Mapping Kolom per Tahun

| Key | 2022 | 2023 | 2024 | 2025 |
|-----|------|------|------|------|
| `c_prov` | R101 | R101 | R101 | PROP |
| `c_kab` | R102 | R102 | R102 | KAB |
| `c_sch` | R610 | R610 | R610 | R611 |
| `c_stat` | R706 | R707 | R707 | R706 |
| `c_roof` | R1806 | R1806 | R1806A | R1606 |
| `asset_prefix` | R2001 | R2001 | R2001 | R1801 |

Tabel di atas cuma sebagian (yg paling beda antar tahun). Mapping lengkap ada di `COLUMN_MAPPING` dlm `preprocess.py`.

Buat tahun baru yg belum ada di `COLUMN_MAPPING`, semua key wajib diisi pakai `--map` (kalo cuma sebagian, bakal error karena key yg lain ga ketemu).

## Catatan Modelling

- Predictor list ada di `PREDICTORS` dlm `modelling.py` (69 fitur)
- Pemekaran wilayah: kode kabupaten 2022 yg kena pemekaran (Riau 1472, Papua) otomatis di-remap ke kode baru lewat `KODE_MAP`
- Filter `--provinsi 94` otomatis ekspansi ke 95/96/97 (Papua mekar), `91` ekspansi ke 91/92 (Papua Barat mekar) - liat `PROV_EXPAND`
