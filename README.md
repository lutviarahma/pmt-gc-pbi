# PMT Susenas - Preprocessing & Modelling

Pipeline preprocessing dan modelling untuk data Susenas PMT, disesuaikan dengan Kuesioner Sensus Ekonomi (SE) 2026.

---

## File

| File | Fungsi |
|------|--------|
| `preprocess.py` | Pipeline preprocessing lengkap (load dataset sampai output `susenas_clean_{year}.csv`) |
| `modelling.py` | Pipeline modelling XGBoost, input dari hasil `preprocess.py` |
| `run_preprocess.py` | Interface terpusat — tinggal lempar data + tahun, otomatis diarahkan ke pipeline yang sesuai |

---

## Alur Data

```
Skenario 1 (KOR + KP terpisah):
KOR + KP -> join -> raw_ruta + individu -> merge -> preprocessing -> susenas_clean_{year}.csv

Skenario 2 (ruta + individu terpisah):
raw_ruta + individu -> merge -> preprocessing -> susenas_clean_{year}.csv

Skenario 3 (sudah merged):
data_merged -> preprocessing -> susenas_clean_{year}.csv
```

---

## run_preprocess.py

Interface terpusat. User tinggal tentukan skenario, masukkan file dan tahun.

**Skenario 3 — data sudah merged (2023, 2024):**
```bash
python run_preprocess.py merged \
  --tahun 2023 2024 \
  --file 23_susenas_kor.csv 24_susenas_kor.csv \
  --output ./hasil
```

**Skenario 1 — KOR + KP terpisah, ada data individu (2025):**
```bash
python run_preprocess.py kor \
  --tahun 2025 \
  --kor 25_susenas_individu.csv \
  --kp 25_susenas_clean.csv \
  --individu 25_ind.csv \
  --output ./hasil
```

**Skenario 1 — KOR + KP terpisah, tanpa individu:**
```bash
python run_preprocess.py kor \
  --tahun 2025 \
  --kor 25_susenas_individu.csv \
  --kp 25_susenas_clean.csv \
  --output ./hasil
```

**Skenario 2 — ruta + individu terpisah (2022):**
```bash
python run_preprocess.py ruta \
  --tahun 2022 \
  --ruta 22_ruta.csv \
  --individu 22_ind.csv \
  --output ./hasil
```

**Beberapa tahun sekaligus:**
```bash
python run_preprocess.py merged \
  --tahun 2023 2024 \
  --file 23_susenas_kor.csv 24_susenas_kor.csv
```

**Custom mapping kolom untuk tahun baru:**
```bash
python run_preprocess.py merged \
  --tahun 2026 \
  --file 26_susenas_merged.csv \
  --map c_prov=PROP c_kab=KAB c_sch=R700 c_ijz=R614
```

---

## preprocess.py

Bisa juga dijalankan langsung per tahun dengan kontrol lebih detail.

**Skenario 3 — langsung ke preprocessing:**
```bash
python preprocess.py --tahun 2023 --merged 23_susenas_kor.csv
python preprocess.py --tahun 2024 --merged 24_susenas_kor.csv
```

**Skenario 1 — KOR + KP terpisah + individu:**
```bash
python preprocess.py --tahun 2025 \
  --kor 25_susenas_individu.csv \
  --kp 25_susenas_clean.csv \
  --individu 25_ind.csv
```

**Skenario 2 — ruta + individu terpisah:**
```bash
python preprocess.py --tahun 2022 \
  --ruta 22_ruta.csv \
  --individu 22_ind.parquet
```

**Filter provinsi + custom chunk size:**
```bash
python preprocess.py --tahun 2025 --merged data.csv --provinsi 34 31 --chunk 50000
```

**Custom mapping kolom:**
```bash
python preprocess.py --tahun 2026 --merged data.csv \
  --map c_prov=PROP c_kab=KAB c_sch=R700 c_ijz=R614 c_stat=R706
```

**Output:**
```
susenas_clean_{year}.csv
```

---

## modelling.py

Input dari hasil `preprocess.py`. Loop per provinsi otomatis, provinsi di-detect dari data jika tidak diisi.

**Semua provinsi:**
```bash
python modelling.py \
  --data 2022:susenas_clean_2022.csv 2023:susenas_clean_2023.csv 2024:susenas_clean_2024.csv \
  --year_test 2024 \
  --output ./output
```

**Filter provinsi tertentu:**
```bash
python modelling.py \
  --data 2022:susenas_clean_2022.csv 2023:susenas_clean_2023.csv 2024:susenas_clean_2024.csv \
  --year_test 2024 \
  --provinsi 34 31 12 \
  --output ./output
```

**Output:**
```
output/report_modelling_{year_test}.csv
```

---

## Format Input

- File CSV dan Parquet didukung, otomatis terdeteksi
- Separator CSV otomatis terdeteksi (`,` / `;` / `|`)
- Nama kolom otomatis dinormalisasi (uppercase, karakter aneh dihapus)

---

## Mapping Kolom per Tahun

| Key | 2022 | 2023 | 2024 | 2025 |
|-----|------|------|------|------|
| `c_prov` | R101 | R101 | R101 | PROP |
| `c_kab` | R102 | R102 | R102 | KAB |
| `c_sch` | R610 | R610 | R610 | R611 |
| `c_stat` | R706 | R707 | R707 | R706 |
| `c_roof` | R1806 | R1806 | R1806A | R1606 |
| `asset_prefix` | R2001 | R2001 | R2001 | R1801 |

Untuk tahun baru yang belum ada di mapping, gunakan `--map` untuk override kolom yang berbeda saja. Kolom yang tidak di-override akan diambil dari tahun terdekat jika ada, atau wajib diisi semua jika tahunnya benar-benar baru.
