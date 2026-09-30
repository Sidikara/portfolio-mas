# Portfolio Optimization Dalam Sistem Perdagangan Multi-Agen

Sistem multi-agen yang mengalokasikan modal ke beberapa agen strategi trading
berdasarkan return harapan, matriks kovarians, dan batasan risiko.

Dikerjakan untuk mata kuliah **Agen Cerdas Enterprise (ACE)**, Program Magister
Kecerdasan Artifisial, FMIPA, Universitas Gadjah Mada.

---

## Gambaran

Program ini menjawab satu pertanyaan: **dari modal yang tersedia, berapa porsi
untuk masing-masing agen strategi?**

Jawabannya berupa tiga angka, yaitu bobot modal untuk ketiga agen. Seluruh kode
bekerja untuk menghasilkan tiga angka tersebut.

```
harga CSV  ->  sinyal agen  ->  statistik  ->  bobot modal  ->  penilaian risiko
 (T, 2)        (T-280, 3)      mu sigma rho    3 angka          HHI, korelasi
```

## Arsitektur tiga tier

| Tier | Peran | Berkas |
|---|---|---|
| 1 | Agen strategi menghasilkan sinyal dari harga | `core/agents.py` |
| 2 | Koordinator mengalokasikan modal antar agen | `core/optimizer.py` |
| 3 | Pemantau risiko konsentrasi | `core/risk.py` |

## Tiga agen strategi

| Agen | Strategi | Sinyal |
|---|---|---|
| 1 | Trend-Following | SMA-50 vs SMA-200 pada BTC |
| 2 | Mean-Reversion | z-score harga BTC terhadap rata-rata bergerak |
| 3 | Market-Neutral | pair trading spread BTC-ETH |

Posisi pada hari `t` ditentukan dari informasi sampai `t`; return direalisasikan
pada `t+1` untuk mencegah *look-ahead bias*.

---

## Instalasi

```bash
git clone https://github.com/<username>/portfolio-mas.git
cd portfolio-mas
pip install -r requirements.txt
```

Membutuhkan Python 3.10 atau lebih baru.

## Penggunaan

```bash
# 1. Unduh data harga BTC/ETH dari Binance
python fetch_data.py --days 730

# 2. Jalankan aplikasi
streamlit run app.py
```

Kedua perintah dijalankan dari folder proyek.

### Opsi fetch_data.py

```bash
python fetch_data.py --days 90                      # 90 hari
python fetch_data.py --days 1000                    # maksimum satu permintaan
python fetch_data.py --out data/harga_saya.csv      # nama berkas lain
```

Bila Binance tidak dapat diakses (di sebagian jaringan Indonesia diblokir pada
tingkat DNS), ganti DNS ke `1.1.1.1`, atau unduh CSV secara manual dan letakkan
di folder `data/` dengan format yang dijelaskan pada `data/README.md`.

---

## Struktur proyek

```
portfolio-mas/
├── app.py                  antarmuka Streamlit, 5 tab
├── fetch_data.py           pengunduh data dari Binance
├── requirements.txt
├── data/                   berkas CSV harga (tidak di-commit)
└── core/
    ├── binance_api.py      pengunduh data Binance
    ├── datasets.py         baca CSV dan hitung statistik agen
    ├── agents.py           TIER 1 - tiga agen strategi
    ├── statistics.py       Formula 1-2: kovarians, return, volatilitas
    ├── optimizer.py        TIER 2 - Formula 3-4: mean-variance, CVaR
    ├── risk.py             TIER 3 - Formula 5: HHI dan konsentrasi
    └── scenarios.py        perbandingan skenario S1-S4
```

Seluruh berkas diberi komentar penjelasan dalam bahasa Indonesia, termasuk
alasan di balik keputusan teknis yang diambil.

---

## Empat skenario alokasi

| Skenario | Dasar | Perlu estimasi `mu`? |
|---|---|---|
| S1 Equal-weight | 1/N untuk semua | tidak |
| S2 Naive | bobot tetap dari laporan | tidak |
| S3 Max-Sharpe | maksimalkan return per unit risiko | **ya** |
| S4 Min-Variance | minimalkan volatilitas portofolio | tidak |

S3 dihilangkan secara otomatis bila tidak ada agen dengan return harapan
positif, karena memaksimalkan Sharpe tidak bermakna ketika pembilangnya sendiri
negatif. S4 tetap valid dalam kondisi itu karena hanya bersandar pada matriks
kovarians (lihat Merton 1980 mengenai kesulitan estimasi `mu` dibanding `Sigma`).

Tabel skenario juga memuat referensi alokasi 100% pada satu agen, dan benchmark
*buy-and-hold* sebagai pembanding.

---

## Temuan

Pengujian pada data Binance nyata di tiga periode berbeda menunjukkan **tidak
ada agen yang menghasilkan alpha**:

| Periode | Agen 1 | Agen 2 | Agen 3 |
|---|---|---|---|
| 90 hari | +0.08 | −0.08 | −0.32 |
| 730 hari | 0.00 | −0.02 | −0.03 |
| 1000 hari | −0.02 | +0.01 | −0.05 |

*(Sharpe ratio harian)*

Hasil ini wajar: SMA crossover dan z-score adalah strategi buku teks yang sudah
dikenal luas selama puluhan tahun.

Karena itu klaim proyek ini tidak berada pada dimensi return, melainkan pada
dimensi risiko: **S4 Min-Variance menurunkan volatilitas portofolio sekitar
8–23% dibanding agen tunggal**, dengan pembanding sengaja dipilih agen tunggal
ber-volatilitas terendah.

---

## Catatan implementasi

**Optimisasi memakai SciPy, bukan CVXPY.** SLSQP untuk masalah kuadratik, HiGHS
untuk program linear. CVXPY memicu *access violation* (exit code `-1073741819`)
pada sebagian instalasi Windows ketika diimpor setelah numpy dan pandas. Karena
seluruh masalah di sini konveks, hasilnya identik dan jaminan optimum global
tetap berlaku — jaminan itu berasal dari bentuk masalahnya, bukan dari solver.

**Beta pada agen pair trading diestimasi dari return, bukan harga.** Regresi
bergulir atas harga yang bertren menghasilkan hedge ratio sangat tidak stabil
(teramati berayun 0.75–1.87 padahal nilai sebenarnya 1.15), karena harga
bersifat non-stasioner.

**Jendela estimasi beta dipisahkan dari jendela sinyal.** Hedge ratio perlu
bergerak lambat agar stabil (250 hari), sinyal perlu responsif (30 hari).

**Biaya transaksi 5 bps** per perubahan posisi, dikenakan hanya saat posisi
berubah.

**SRI penuh belum diimplementasikan.** Komponen centrality dan spillover
berbasis Granger causality memerlukan deret waktu jauh lebih panjang.
Digantikan proxy berbasis HHI dan korelasi tertimbang, yang keduanya memiliki
dasar literatur.

---

## Referensi

- Markowitz, H. (1952). Portfolio Selection. *The Journal of Finance*, 7(1).
- Merton, R. C. (1980). On estimating the expected return on the market.
  *Journal of Financial Economics*, 8(4).
- Rockafellar, R. T., & Uryasev, S. (2000). Optimization of Conditional
  Value-at-Risk. *Journal of Risk*, 2(3).
- Billio, M., Getmansky, M., Lo, A. W., & Pelizzon, L. (2012). Econometric
  measures of connectedness and systemic risk. *Journal of Financial
  Economics*, 104(3).

---

## Lisensi

MIT. Lihat berkas `LICENSE`.
