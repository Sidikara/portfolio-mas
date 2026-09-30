"""
===============================================================================
 risk.py  -  TIER 3: PEMANTAU RISIKO KONSENTRASI
===============================================================================

APA ISI BERKAS INI
    Ukuran-ukuran untuk MENILAI portofolio yang sudah terpilih.

    CATATAN PENTING: berkas ini TIDAK MENGUBAH KEPUTUSAN APA PUN. Bobot sudah
    ditentukan optimizer.py; di sini hanya dinilai seberapa terkonsentrasi
    dan seberapa saling terkait alokasinya.

POSISI DALAM ALUR PROGRAM
    optimizer.py  ->  [ risk.py ]  ->  ditampilkan di Tab 4
      cari bobot       nilai bobot

LIMA UKURAN UTAMA
    HHI                    konsentrasi modal
    Effective N            berapa agen yang "benar-benar" memegang modal
    Weighted correlation   rata-rata korelasi, ditimbang bobot
    Diversification ratio  seberapa besar manfaat diversifikasi
    CRI                    proxy risiko sistemik (pengganti SRI penuh)

BATASAN
    SRI penuh yang direncanakan di laporan, yang mencakup centrality dan
    spillover berbasis Granger causality, BELUM diimplementasikan. Komponen
    itu butuh deret waktu panjang dan kalibrasi bobot tersendiri. Sebagai
    gantinya dipakai CRI, proxy berbasis HHI dan korelasi tertimbang.
===============================================================================
"""

import numpy as np

from .statistics import portfolio_volatility


def herfindahl(w: np.ndarray) -> float:
    """
    KEGUNAAN  Mengukur KONSENTRASI modal. Makin tinggi, makin terpusat.

    MASUK     w bentuk (3,)
    KELUAR    satu angka antara 1/n dan 1

    RUMUS     HHI = jumlah dari w_i kuadrat

    CONTOH
        merata (1/3, 1/3, 1/3)   ->  3 x (1/3)^2       = 0.333  (paling aman)
        (0.26, 0.54, 0.20)       ->  0.0676+0.2916+0.04 = 0.399
        satu agen (1, 0, 0)      ->  1^2                = 1.000  (paling rapuh)

    Operator ** pada array numpy memangkatkan SETIAP ELEMEN, bukan array
    secara keseluruhan.
    """
    w = np.asarray(w, dtype=float)
    return float(np.sum(w**2))


def effective_n_agents(w: np.ndarray) -> float:
    """
    KEGUNAAN  Menerjemahkan HHI jadi angka yang lebih mudah dibaca:
              "berapa agen yang sebenarnya memegang modal".

    RUMUS     1 / HHI

    CONTOH    HHI 0.333 -> 3.00 agen  (modal tersebar ke tiga agen)
              HHI 0.399 -> 2.51 agen  (setara menyebar ke 2.5 agen)
              HHI 1.000 -> 1.00 agen  (semua modal di satu agen)
    """
    hhi = herfindahl(w)
    # float("inf") adalah tak hingga, dipakai sebagai penanda kalau HHI nol
    # (yang seharusnya tidak pernah terjadi karena jumlah bobot selalu 1)
    return float("inf") if hhi <= 0 else 1.0 / hhi


def weighted_correlation(w: np.ndarray, rho: np.ndarray) -> float:
    """
    KEGUNAAN  Menghitung rata-rata korelasi antar agen, DITIMBANG oleh bobot
              modalnya. Korelasi antara dua agen berbobot besar lebih
              berpengaruh daripada antara dua agen berbobot kecil.

    MASUK     w (3,), rho (3,3)
    KELUAR    satu angka

    PASANGAN YANG DIKUNJUNGI bila n = 3:
        i=0 : j=1, j=2      -> pasangan (0,1) dan (0,2)
        i=1 : j=2           -> pasangan (1,2)
        i=2 : (kosong)      -> range(3,3) tidak menghasilkan apa-apa
        Total 3 pasangan, bukan 9.

    Kalau range(i + 1, n) diubah jadi range(n), setiap pasangan dihitung DUA
    KALI dan pasangan (i,i) yang nilainya selalu 1 ikut terhitung, sehingga
    hasilnya salah.
    """
    w = np.asarray(w, dtype=float)
    rho = np.asarray(rho, dtype=float)
    n = len(w)

    num = 0.0  # penjumlah pembilang
    den = 0.0  # penjumlah penyebut

    for i in range(n):
        # range(i + 1, n) memastikan setiap pasangan dikunjungi TEPAT SEKALI
        for j in range(i + 1, n):
            weight = w[i] * w[j]        # bobot pasangan ini
            num += weight * rho[i, j]   # += berarti "tambahkan ke nilai sekarang"
            den += weight

    # Penjagaan pembagian nol
    return float(num / den) if den > 1e-12 else 0.0


def diversification_ratio(w: np.ndarray, sigma: np.ndarray, cov: np.ndarray) -> float:
    """
    KEGUNAAN  Mengukur seberapa besar manfaat diversifikasi yang didapat.

    RUMUS     (jumlah w_i * sigma_i) / sigma_portofolio

              Pembilang  = volatilitas SEANDAINYA tidak ada manfaat
                           diversifikasi sama sekali
              Penyebut   = volatilitas portofolio yang sebenarnya

    CARA BACA
        = 1    tidak ada manfaat diversifikasi
        > 1    makin besar makin baik
        1.35   berarti risiko portofolio 35 persen lebih rendah daripada
               penjumlahan naif
    """
    w = np.asarray(w, dtype=float)
    weighted_vol = float(w @ np.asarray(sigma))
    port_vol = portfolio_volatility(w, cov)
    return weighted_vol / port_vol if port_vol > 1e-12 else 1.0


def concentration_risk_index(
    w: np.ndarray,
    rho: np.ndarray,
    weight_hhi: float = 0.5,
    weight_corr: float = 0.5,
) -> float:
    """
    KEGUNAAN  Proxy sederhana untuk risiko sistemik, menggantikan SRI penuh.

    RUMUS     CRI = a * HHI_ternormalisasi + b * korelasi_tertimbang

    MASUK     w (3,), rho (3,3), dua bobot yang bisa digeser dari sidebar
    KELUAR    satu angka antara 0 dan 1

    BATASAN YANG PERLU DISEBUT SAAT PRESENTASI
        Bobot 0.5 dan 0.5 adalah ASUMSI, bukan hasil kalibrasi. Keduanya bisa
        digeser dari sidebar untuk menguji sensitivitasnya.
    """
    w = np.asarray(w, dtype=float)
    n = len(w)
    hhi = herfindahl(w)

    # Normalkan HHI ke rentang 0 sampai 1.
    # Nilai terkecil yang mungkin adalah 1/n, terbesar adalah 1.
    # Rumus (x - min) / (max - min) memetakan keduanya jadi 0 dan 1.
    hhi_norm = (hhi - 1.0 / n) / (1.0 - 1.0 / n) if n > 1 else 1.0

    # Korelasi negatif dianggap nol, karena korelasi negatif justru
    # MENGURANGI risiko sistemik dan tidak semestinya menambah indeks.
    corr = max(weighted_correlation(w, rho), 0.0)

    total = weight_hhi + weight_corr
    if total <= 0:
        return 0.0
    return float((weight_hhi * hhi_norm + weight_corr * corr) / total)


def risk_report(w: np.ndarray, sigma: np.ndarray, rho: np.ndarray, cov: np.ndarray) -> dict:
    """
    KEGUNAAN  Membungkus kelima ukuran di atas jadi satu panggilan, untuk
              ditampilkan sebagai metrik di Tab 4.

    KELUAR    dict dengan 5 kunci
    """
    return {
        "HHI": herfindahl(w),
        "Effective N": effective_n_agents(w),
        "Weighted correlation": weighted_correlation(w, rho),
        "Diversification ratio": diversification_ratio(w, sigma, cov),
        "CRI": concentration_risk_index(w, rho),
    }


def historical_var_cvar(
    returns: np.ndarray, w: np.ndarray, alpha: float = 0.95
) -> tuple[float, float]:
    """
    KEGUNAAN  Menghitung VaR dan CVaR secara EMPIRIS dari data historis.

    BEDANYA DENGAN cvar_optimize DI optimizer.py
        Di optimizer, CVaR dipakai sebagai FUNGSI TUJUAN untuk MENCARI bobot.
        Di sini, VaR dan CVaR dihitung dari bobot yang SUDAH terpilih,
        sebagai pelaporan. Peran keduanya berbeda.

    MASUK     returns (T,3), w (3,), alpha
    KELUAR    tuple berisi dua angka, keduanya sebagai kerugian POSITIF

    BEDA VaR DAN CVaR
        VaR  menjawab: seberapa buruk pada batas 5 persen terburuk
        CVaR menjawab: seberapa buruk RATA-RATANYA ketika batas itu terlampaui

    CONTOH
        losses       : [-1, 0, 2, 5, 8, 3, -2, 1, 9, 4]
        quantile 0.9 : var = 8.1
        losses >= 8.1: hanya [9]
        cvar         : 9.0
    """
    # Matriks (T,3) @ array (3,) -> array (T,) berisi return portofolio harian
    port = np.asarray(returns, dtype=float) @ np.asarray(w, dtype=float)

    # Tanda minus membalik tanda setiap elemen, sehingga kerugian jadi
    # bilangan positif dan lebih mudah dibaca
    losses = -port

    # np.quantile(arr, 0.95) mencari nilai yang melampaui 95 persen elemen
    var = float(np.quantile(losses, alpha))

    # losses[losses >= var] adalah BOOLEAN INDEXING: pilih hanya elemen yang
    # memenuhi kondisi. PERHATIKAN panjang array BERUBAH di sini, berbeda
    # dari kebanyakan operasi lain yang mempertahankan panjang.
    tail = losses[losses >= var]

    # tail.size adalah jumlah elemen. Kalau kosong, mean() menghasilkan NaN,
    # jadi dipakai var sebagai cadangan.
    cvar = float(tail.mean()) if tail.size else var
    return var, cvar


def max_drawdown(returns: np.ndarray, w: np.ndarray) -> float:
    """
    KEGUNAAN  Menghitung penurunan TERDALAM dari puncak kurva ekuitas.
              Mengukur kerugian terburuk yang dialami seseorang yang masuk
              pada waktu paling sial.

    MASUK     returns (T,3), w (3,)
    KELUAR    satu angka dalam persen

    CONTOH
        port     : [10, -20, 5]                  <- persen
        equity   : [1.10, 0.88, 0.924]           <- cumprod
        peak     : [1.10, 1.10, 1.10]            <- cummax
        turun    : [0.0, 0.20, 0.16]
        hasil    : 20.0 persen
    """
    port = np.asarray(returns, dtype=float) @ np.asarray(w, dtype=float)

    # cumprod mengalikan kumulatif, membangun kurva ekuitas dari faktor
    # pertumbuhan harian
    equity = np.cumprod(1.0 + port / 100.0)

    # maximum.accumulate mencatat nilai tertinggi sampai posisi itu
    peak = np.maximum.accumulate(equity)

    return float(np.max((peak - equity) / peak) * 100.0)
