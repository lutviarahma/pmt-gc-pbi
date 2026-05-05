
---

# PMT DTSEN GC PBI FASIH 2026 – Preprocessing Pipeline

Pipeline preprocessing end-to-end untuk data **DTSEN GC PBI FASIH 2026**, mencakup integrasi multi-source, cleaning, imputasi, hingga feature engineering untuk menghasilkan dataset siap modelling.

---

## File

* `prepocessing gc pbi fasih gianyar.py` → pipeline utama (Stage 1–3)
* Output:

  * `dtsen_clean_lite_2026_gianyar.csv`


---

##  Alur Data

```
Meteran + Root + AK
        ↓
[Stage 1] Cleaning & Filtering
        ↓
[Stage 2] Imputation + Chunk Processing
        ↓
[Stage 3] Feature Engineering (Household Level)
        ↓
dtsen_clean_lite_2026.csv
```

---

## Pipeline

### Stage 1 – Integrasi & Filtering

**Input:**

* `meteran_listrik_*.csv`
* `ak_nested_*.csv`
* `root_table_*.csv`

**Proses:**

* Normalisasi nama kolom
* Filter rumah tanpa `jenis_lantai`
* Filter individu:

  * `2` → meninggal
  * `6` → tidak ditemukan
  * `7` → pindah
* Agregasi listrik (`daya_maks`)

**Merge:**

```
root + meteran → ruta
ruta + AK → individu
```

**Output sementara:**

```
temp_merged.csv
```

---

### Stage 2 – Imputasi & Chunk Processing

**Tujuan:**

* Handle big data (chunking)
* Hindari split rumah tangga (tail buffer)

**Proses:**

* Konversi numerik (`umur`, `sekolah`)
* Imputasi:

  * umur 0–4 → sekolah = 0
* Drop:

  * umur null
  * sekolah null / 9

**Output:**

```
2026_dtsen_merged_imputed_gianyar.csv
```

---

### Stage 3 – Feature Engineering

Transformasi dari **level individu → rumah tangga**

#### Fitur Individu

* Pendidikan → `h_ngrad_*`
* Status sekolah
* Status kawin
* Gender
* Kategori umur
* Status kerja

#### Agregasi Rumah Tangga

* `h_hhcount`
* Pivot ke wide format

#### Fitur Rumah Tangga

* Luas lantai + log transform
* Aset:

  * AC, kulkas, motor, mobil, dll
* Internet:

  * berdasarkan pengeluaran pulsa

#### Kategori Rumah (One-hot)

* House
* Floor
* Wall
* Roof
* Water
* Electricity
* Toilet

#### Finalisasi

* Normalisasi pendidikan
* Tambah:

  * `h_nfamily = 1`
  * `kode_prov`
  * `kode_kab`

**Output akhir:**

```
dtsen_clean_lite_2026_gianyar.csv
```

---

## Cara Menjalankan

```bash
python prepocessing gc pbi fasih gianyar.py
```

---

## Konfigurasi

```python
PATH_METERAN = "meteran_listrik_202604080756.csv"
PATH_AK = "ak_nested_202604071642_wo_nik.csv"
PATH_ROOT = "root_table_202604071643_wo_nik.csv"

CHUNK_SIZE = 100000
```

---

## Output

| File                                    | Deskripsi                 |
| --------------------------------------- | ------------------------- |
| `2026_dtsen_merged_imputed_gianyar.csv` | hasil cleaning + imputasi |
| `dtsen_clean_lite_2026_gianyar.csv`             | dataset final             |

---





