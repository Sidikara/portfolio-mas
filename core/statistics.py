"""
===============================================================================
 statistics.py  -  PERHITUNGAN STATISTIK PORTOFOLIO
===============================================================================

APA ISI BERKAS INI
    Rumus dasar yang dipakai koordinator (optimizer.py) untuk menilai sebuah
    portofolio. Semua fungsi di sini murni matematika, tidak ada keputusan.

POSISI DALAM ALUR PROGRAM
    agents.py  ->  datasets.py  ->  [ statistics.py ]  ->  optimizer.py
                                     hitung kovarians     cari bobot

DUA RUMUS UTAMA (Formula 1 dan 2 pada laporan)
    Formula 1:  E[Rp]   = w' mu                    <- portfolio_return
    Formula 2:  sigma_p = akar(w' Sigma w)         <- portfolio_volatility

ARTI LAMBANG
    w      = bobot modal tiap agen, contoh [0.26, 0.54, 0.20], jumlahnya 1
    mu     = return harapan tiap agen, contoh [2.05, 1.02, 0.52]
    sigma  = volatilitas tiap agen, contoh [0.158, 0.079, 0.079]
    rho    = matriks korelasi 3x3, diagonalnya selalu 1
    Sigma  = matriks kovarians 3x3, dihitung dari sigma dan rho
===============================================================================
"""

import numpy as np


def covariance_matrix(sigma: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """
    KEGUNAAN
        Menggabungkan volatilitas (sigma) dan korelasi (rho) menjadi satu
        matriks kovarians (Sigma). Matriks inilah masukan utama optimizer.

    MASUK   sigma bentuk (3,)   -> [2.2, 1.8, 0.7]
            rho   bentuk (3,3)  -> [[1, 0.35, 0.2], [0.35, 1, 0.15], ...]
    KELUAR  Sigma bentuk (3,3)

    RUMUS   Sigma[i][j] = rho[i][j] * sigma[i] * sigma[j]

    DIPANGGIL OLEH  app.py, verify, dan seluruh pengujian
    """
    # asarray: pastikan bertipe array numpy, sehingga fungsi ini tetap jalan
    # meskipun dipanggil dengan list Python biasa seperti [2.2, 1.8, 0.7]
    sigma = np.asarray(sigma, dtype=float)
    rho = np.asarray(rho, dtype=float)

    # np.outer(a, b) membuat matriks berisi a[i] * b[j] untuk setiap pasangan.
    #   np.outer([2, 3], [2, 3]) -> [[4, 6],
    #                                [6, 9]]
    # Tanda * di sini adalah perkalian ELEMEN PER ELEMEN, bukan perkalian
    # matriks. Kalau diganti @ hasilnya salah total.
    cov = rho * np.outer(sigma, sigma)

    # Matriks kovarians secara teori pasti simetris (elemen [i][j] sama dengan
    # [j][i]). Tetapi pembulatan desimal kadang membuat keduanya berbeda pada
    # digit terakhir, dan solver bisa menolak matriks seperti itu.
    # cov.T adalah transpose. Merata-ratakan cov dengan transposenya memaksa
    # hasilnya simetris sempurna.
    return 0.5 * (cov + cov.T)


def shrink_covariance(cov: np.ndarray, intensity: float = 0.0) -> np.ndarray:
    """
    KEGUNAAN
        Menstabilkan matriks kovarians ketika jumlah hari data sedikit.
        Caranya menarik Sigma ke arah matriks diagonal (korelasi diabaikan).

    MASUK   cov bentuk (3,3), intensity antara 0 dan 1
    KELUAR  matriks (3,3)

    EFEK    intensity = 0   -> kovarians apa adanya
            intensity = 0.5 -> elemen di luar diagonal dikecilkan separuh
            intensity = 1   -> korelasi hilang sama sekali

    DIKENDALIKAN OLEH  slider "Shrinkage kovarians" di sidebar

    CATATAN Nilai 1.0 membuat volatilitas portofolio terlihat lebih kecil,
            tetapi itu menyesatkan karena informasi korelasinya dibuang.
            Untuk data pendek, nilai wajar adalah 0.3 sampai 0.5.
    """
    # Jalan pintas: kalau tidak ada shrinkage, kembalikan apa adanya
    if intensity <= 0:
        return cov

    # np.diag punya DUA perilaku berbeda:
    #   diberi matriks  -> ambil diagonalnya jadi array 1D    [[4,3],[3,9]] -> [4,9]
    #   diberi array 1D -> buat matriks diagonal              [4,9] -> [[4,0],[0,9]]
    # Jadi np.diag(np.diag(cov)) = matriks yang hanya berisi diagonal cov,
    # sisanya nol. Ini yang disebut "target" pada metode shrinkage.
    target = np.diag(np.diag(cov))

    # Campur dua matriks. Diagonalnya tidak berubah (karena sama di kedua
    # matriks), hanya elemen di luar diagonal yang mengecil.
    return (1.0 - intensity) * cov + intensity * target


def portfolio_return(w: np.ndarray, mu: np.ndarray) -> float:
    """
    KEGUNAAN  Formula 1. Menghitung return harapan portofolio.
    MASUK     w bentuk (3,), mu bentuk (3,)
    KELUAR    satu angka (float)

    CONTOH    w  = [0.5, 0.5]
              mu = [2.0, 4.0]
              hasil = 0.5*2.0 + 0.5*4.0 = 3.0
    """
    # Operator @ pada dua array 1D menghitung jumlah dari perkalian elemen,
    # yaitu w[0]*mu[0] + w[1]*mu[1] + w[2]*mu[2].
    # float(...) mengubah hasil numpy menjadi angka Python biasa.
    return float(np.asarray(w) @ np.asarray(mu))


def portfolio_volatility(w: np.ndarray, cov: np.ndarray) -> float:
    """
    KEGUNAAN  Formula 2. Menghitung volatilitas (risiko) portofolio.
    MASUK     w bentuk (3,), cov bentuk (3,3)
    KELUAR    satu angka (float)

    DI SINILAH MANFAAT DIVERSIFIKASI BEKERJA.
        Return portofolio adalah rata-rata tertimbang biasa, jadi sifatnya
        MENJUMLAH. Volatilitas tidak begitu: hasil w' Sigma w memuat suku
        silang berisi korelasi. Kalau korelasi antar agen rendah, suku silang
        itu kecil, sehingga risiko gabungan LEBIH KECIL daripada rata-rata
        risiko masing-masing agen.

    CONTOH    w   = [0.5, 0.5]
              cov = [[4, 3],
                     [3, 9]]
              cov @ w        = [3.5, 6.0]
              w @ [3.5, 6.0] = 4.75          <- ini variance
              akar(4.75)     = 2.179         <- ini volatilitas
    """
    w = np.asarray(w, dtype=float)

    # w @ cov @ w dikerjakan dari kiri: (w @ cov) dulu, hasilnya array (3,),
    # lalu @ w lagi, hasilnya satu angka.
    variance = float(w @ np.asarray(cov) @ w)

    # max(variance, 0.0) mencegah akar bilangan negatif. Perhitungan desimal
    # kadang menghasilkan -0.0000000001 padahal maksudnya nol, dan
    # np.sqrt(bilangan negatif) menghasilkan NaN yang merusak seluruh tabel.
    return float(np.sqrt(max(variance, 0.0)))


def sharpe_ratio(w: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf: float = 0.0) -> float:
    """
    KEGUNAAN  Menghitung Sharpe ratio = return dibagi risiko.
              Ini "nilai rapor" portofolio: berapa untung per satuan risiko.

    MASUK     w (3,), mu (3,), cov (3,3), rf = bunga bebas risiko
    KELUAR    satu angka (float)

    CATATAN   rf (risk-free rate) diisi 0 sesuai laporan.
    """
    vol = portfolio_volatility(w, cov)

    # Penjagaan pembagian nol. Perbandingan dilakukan terhadap 1e-12
    # (seper satu triliun), bukan terhadap 0, karena hasil perhitungan
    # desimal jarang tepat nol.
    if vol <= 1e-12:
        return 0.0

    return (portfolio_return(w, mu) - rf) / vol


def portfolio_metrics(
    w: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf: float = 0.0
) -> dict:
    """
    KEGUNAAN  Membungkus tiga perhitungan di atas jadi satu panggilan, supaya
              pemanggil tidak perlu memanggil tiga fungsi terpisah.

    MASUK     w (3,), mu (3,), cov (3,3)
    KELUAR    dict berisi 4 kunci:
                "weights"    -> array bobot
                "return"     -> float
                "volatility" -> float
                "sharpe"     -> float

    KENAPA DICT
        Karena dict bisa DITAMBAH kunci baru setelahnya. scenarios.py
        memanfaatkan ini: setelah memanggil fungsi ini, ia menambahkan
        kunci "label", "kelompok", "hhi", dan "drawdown" ke dict yang sama.

    DIPANGGIL OLEH  seluruh fungsi di optimizer.py dan scenarios.py
    """
    ret = portfolio_return(w, mu)
    vol = portfolio_volatility(w, cov)
    return {
        "weights": np.asarray(w, dtype=float),
        "return": ret,
        "volatility": vol,
        # sharpe dihitung ulang di sini (bukan memanggil sharpe_ratio) supaya
        # tidak menghitung vol dua kali
        "sharpe": 0.0 if vol <= 1e-12 else (ret - rf) / vol,
    }


def condition_number(cov: np.ndarray) -> float:
    """
    KEGUNAAN  Memeriksa "kesehatan" matriks kovarians.
              Nilai besar = matriks hampir singular = hasil optimisasi rapuh.

    MASUK     cov bentuk (3,3)
    KELUAR    satu angka (float), atau tak hingga bila matriks rusak

    CARA BACA
        < 100    : sehat
        100-1e4  : masih wajar
        > 1e4    : matriks hampir singular, biasanya karena jumlah hari
                   terlalu sedikit. Aplikasi menampilkan peringatan kuning.

    CONTOH    eig = [0.02, 1.5, 8.3]  ->  8.3 / 0.02 = 415
    """
    # eigvalsh khusus untuk matriks simetris, lebih cepat dan stabil
    # daripada eigvals biasa. Hasilnya array 1D berisi nilai eigen.
    eig = np.linalg.eigvalsh(cov)

    lo = float(np.min(eig))

    # Nilai eigen negatif atau nol berarti matriksnya cacat. float("inf")
    # adalah bilangan tak hingga, dipakai sebagai penanda kondisi terburuk.
    if lo <= 0:
        return float("inf")

    return float(np.max(eig) / lo)
