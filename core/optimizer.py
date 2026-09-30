"""
===============================================================================
 optimizer.py  -  TIER 2: KOORDINATOR ALOKASI MODAL
===============================================================================

APA ISI BERKAS INI
    Mesin yang menghitung BOBOT MODAL: berapa persen untuk tiap agen.
    Ini berkas paling matematis di seluruh program.

POSISI DALAM ALUR PROGRAM
    agents.py -> datasets.py -> statistics.py -> [ optimizer.py ] -> risk.py
                                                  cari bobot w

EMPAT FUNGSI UTAMA
    mean_variance  Formula 3. Seimbangkan return dan risiko, diatur lambda
    max_sharpe     Cari Sharpe tertinggi, tanpa perlu menebak lambda
    min_variance   Cari risiko terendah. TIDAK memakai mu sama sekali
    cvar_optimize  Formula 4. Minimalkan kerugian pada ekor distribusi

KENAPA TIDAK MEMAKAI CVXPY
    Rencana awal memakai CVXPY. Tetapi pada Windows yang dipakai tim,
    mengimpor CVXPY setelah numpy dan pandas memicu access violation dengan
    kode keluar -1073741819, yaitu kegagalan tingkat DLL, bukan kesalahan
    kode Python. Karena seluruh masalah di sini KONVEKS, SLSQP dari SciPy
    memberi hasil identik (sudah diverifikasi angka demi angka).

KENAPA HASILNYA DIJAMIN OPTIMAL
    Seluruh masalah di sini berbentuk kuadratik konveks dengan kendala
    linear. Fungsi konveks bentuknya seperti MANGKUK: hanya punya satu titik
    terendah. Jadi begitu solver menemukan optimum lokal, itu PASTI optimum
    global. Jaminan ini berasal dari BENTUK MASALAHNYA, bukan dari
    kecanggihan solvernya.
===============================================================================
"""

import numpy as np
from scipy.optimize import linprog, minimize

from .statistics import portfolio_metrics

_MAXITER = 500
_TOL = 1e-12


# =============================================================================
#  FUNGSI PEMBANTU
# =============================================================================

def _starting_points(n: int, w_max: float, n_random: int = 6, seed: int = 0) -> list:
    """
    KEGUNAAN  Menyiapkan beberapa titik awal bagi solver.

    MASUK     n = jumlah agen, w_max = batas bobot maksimum
    KELUAR    list berisi beberapa array bobot, masing-masing berjumlah 1

    KENAPA BANYAK TITIK AWAL
        Karena masalahnya konveks, secara teori SATU titik awal sudah cukup.
        Multi-start di sini hanya pengaman numerik kalau solver tersangkut.
    """
    # Titik awal pertama: alokasi merata, sepertiga untuk masing-masing
    # np.full(n, nilai) membuat array sepanjang n berisi nilai yang sama
    points = [np.full(n, 1.0 / n)]

    # Satu titik awal per agen: pusatkan bobot pada agen itu sampai batas
    # w_max, lalu bagi sisanya rata ke agen lain
    for i in range(n):
        w = np.full(n, (1.0 - min(w_max, 1.0)) / max(n - 1, 1))
        w[i] = min(w_max, 1.0)
        total = w.sum()
        if total > 0:
            points.append(w / total)

    # Beberapa titik acak. Distribusi Dirichlet dipilih karena hasilnya
    # SELALU berjumlah tepat 1, persis seperti syarat bobot portofolio.
    # Argumen seed membuat hasilnya selalu sama tiap kali dijalankan,
    # sehingga hasil program dapat direproduksi.
    rng = np.random.default_rng(seed)
    for _ in range(n_random):
        w = rng.dirichlet(np.ones(n))
        points.append(w)

    # Rapikan: potong ke rentang yang sah, lalu normalkan ulang
    cleaned = []
    for w in points:
        w = np.clip(w, 0.0, w_max)
        total = w.sum()
        if total > 1e-9:
            cleaned.append(w / total)
    return cleaned


def _clean(w: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    """
    KEGUNAAN  Merapikan bobot hasil solver.

    MASALAH   Solver numerik sering menghasilkan angka seperti 0.0000000003
              atau bahkan -0.0000000001, padahal maksudnya NOL.

    CONTOH
        masuk      : [0.3, -0.0000000004, 0.7000001]
        abs < 1e-6 : [ F ,       T       ,    F     ]
        isi nol    : [0.3,      0.0      , 0.7000001]
        clip       : [0.3,      0.0      , 0.7000001]
        sum        : 1.0000001
        keluar     : [0.29999997, 0.0, 0.70000003]
    """
    w = np.asarray(w, dtype=float)

    # np.abs(w) < floor menghasilkan array True/False.
    # w[array_boolean] = 0.0 mengisi 0.0 HANYA di posisi bernilai True.
    w[np.abs(w) < floor] = 0.0

    # np.clip(w, 0.0, None) memotong nilai di bawah 0 jadi 0.
    # None pada batas atas berarti tidak ada batas atas.
    w = np.clip(w, 0.0, None)

    # Normalkan ulang supaya jumlahnya kembali tepat 1 setelah pembulatan
    total = w.sum()
    return w / total if total > 0 else w


def _minimize_simplex(fun, jac, n, w_max, w_min=0.0, seed=0):
    """
    KEGUNAAN  Mencari nilai w yang MEMINIMALKAN fungsi fun, dengan syarat:
                jumlah w = 1
                w_min <= setiap w <= w_max

    MASUK     fun = fungsi tujuan, jac = turunannya, n = jumlah agen
    KELUAR    array bobot (n,)

    DIPANGGIL OLEH  mean_variance dan min_variance
    """
    # Penjagaan logis: kalau w_max = 0.3 dan ada 3 agen, jumlah maksimum yang
    # mungkin hanya 0.9, sehingga syarat "jumlah = 1" mustahil dipenuhi.
    if w_max * n < 1.0 - 1e-9:
        raise ValueError(
            f"w_max={w_max} terlalu kecil untuk {n} agen; minimal {1.0 / n:.4f}."
        )

    # [(a, b)] * n menggandakan tuple sebanyak n kali:
    #   n=3, w_max=0.5  ->  [(0.0, 0.5), (0.0, 0.5), (0.0, 0.5)]
    bounds = [(w_min, w_max)] * n

    # Kendala persamaan. SciPy mensyaratkan fungsi bertipe "eq" bernilai NOL
    # pada solusi. Jadi "np.sum(w) - 1.0 = 0" berarti jumlah w tepat satu.
    # Kalau ditulis "np.sum(w)" saja tanpa dikurangi 1.0, artinya berubah
    # menjadi "jumlah w harus nol" dan hasilnya salah.
    #
    # lambda w: ... adalah cara membuat fungsi kecil tanpa nama dalam satu baris.
    # jac adalah turunan kendala; menyediakannya membuat solver lebih stabil.
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0,
                    "jac": lambda w: np.ones_like(w)}]

    best_w, best_f = None, np.inf
    for w0 in _starting_points(n, w_max, seed=seed):
        res = minimize(
            fun,                  # fungsi yang diminimalkan
            w0,                   # titik awal
            jac=jac,              # turunan, mempercepat dan menstabilkan
            method="SLSQP",       # Sequential Least Squares Programming
            bounds=bounds,        # batas bawah dan atas tiap variabel
            constraints=constraints,
            options={"maxiter": 1000,   # batas iterasi sebelum menyerah
                     "ftol": 1e-10},    # ambang perubahan yang dianggap konvergen
        )
        # res.success -> True kalau solver berhasil
        # res.x       -> array solusi
        # res.fun     -> nilai fungsi tujuan pada solusi itu
        if res.success and res.fun < best_f:
            best_w, best_f = res.x, res.fun

    if best_w is None:
        raise RuntimeError("Optimisasi gagal konvergen dari seluruh titik awal.")
    return best_w


# =============================================================================
#  FORMULA 3: MEAN-VARIANCE
# =============================================================================

def mean_variance(
    mu: np.ndarray,
    cov: np.ndarray,
    lam: float = 1.0,
    w_max: float = 1.0,
    w_min: float = 0.0,
) -> dict:
    """
    KEGUNAAN  Formula 3. Mencari bobot yang menyeimbangkan return dan risiko.

    RUMUS     maksimalkan:  w'mu  -  lambda * w'Sigma w
              dengan syarat: jumlah w = 1,  0 <= w <= w_max

    PERAN LAMBDA (parameter risk aversion)
        lambda KECIL -> risiko hampir tidak dihukum -> kejar return
                        -> sering menghasilkan SOLUSI SUDUT (semua modal
                           ke satu agen)
        lambda BESAR -> risiko dihukum berat -> bobot menyebar
                        -> mendekati portofolio varians minimum

    MASUK     mu (3,), cov (3,3), lam = satu angka
    KELUAR    dict berisi weights, return, volatility, sharpe, lambda, status
    """
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    n = len(mu)

    # Fungsi tujuan dibagi (1 + lam) supaya besarannya tetap wajar pada lambda
    # ekstrem. Membagi dengan konstanta POSITIF tidak memindahkan titik
    # optimum, hanya mengubah skalanya.
    #
    # Tanpa baris ini, pada lambda di atas seratus ribu gradiennya meledak
    # dan solver gagal konvergen.
    skala = 1.0 / (1.0 + lam)

    # Fungsi didefinisikan DI DALAM fungsi lain, sehingga bisa mengakses
    # variabel mu, cov, lam dari luar. Ini disebut CLOSURE.
    def fun(w):
        # Tanda minus di depan (mu @ w) karena SciPy hanya bisa
        # MEMINIMALKAN. Untuk memaksimalkan sesuatu, yang diminimalkan
        # adalah negatifnya.
        return skala * (-(mu @ w) + lam * (w @ cov @ w))

    def jac(w):
        # Turunan analitis dari fun terhadap w:
        #   turunan dari -(mu @ w)        adalah  -mu
        #   turunan dari lam * w'Sigma w  adalah  2 * lam * Sigma @ w
        return skala * (-mu + 2.0 * lam * (cov @ w))

    w = _minimize_simplex(fun, jac, n, w_max, w_min)

    result = portfolio_metrics(_clean(w), mu, cov)
    result["lambda"] = lam
    result["status"] = "optimal"
    return result


# =============================================================================
#  MAX-SHARPE (PORTOFOLIO TANGENCY)
# =============================================================================

def max_sharpe(
    mu: np.ndarray,
    cov: np.ndarray,
    w_max: float = 1.0,
    rf: float = 0.0,
) -> dict:
    """
    KEGUNAAN  Mencari bobot dengan Sharpe ratio TERTINGGI, tanpa perlu
              menebak-nebak nilai lambda.

    MASALAHNYA
        Sharpe ratio berbentuk PECAHAN (return dibagi volatilitas), dan
        bentuk pecahan sulit dioptimasi langsung karena tidak konkaf.

    TRIKNYA: SUBSTITUSI VARIABEL
        Ganti y = w / kappa. Masalahnya berubah jadi kuadratik biasa:
            minimalkan  y' Sigma y
            syarat      (mu - rf)' y = 1,  y >= 0,  y <= w_max * jumlah(y)
        lalu kembalikan: w = y / jumlah(y)

        Setelah substitusi bentuknya kuadratik konveks dengan kendala linear,
        sehingga optimum global terjamin.

    MASUK     mu (3,), cov (3,3), w_max
    KELUAR    dict berisi weights, return, volatility, sharpe, status
    """
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    n = len(mu)
    excess = mu - rf  # kelebihan return di atas bunga bebas risiko

    # np.all(kondisi) bernilai True hanya kalau SELURUH elemen memenuhi.
    # Kalau tidak ada satu pun agen dengan return positif, portofolio
    # Sharpe-maksimum tidak terdefinisi. Pemanggil (scenarios.py) menangani
    # ini dengan tidak menampilkan skenario S3.
    if np.all(excess <= 0):
        raise ValueError("Tidak ada agen dengan excess return positif.")

    def fun(y):
        return y @ cov @ y

    def jac(y):
        return 2.0 * (cov @ y)

    constraints = [
        {"type": "eq", "fun": lambda y: excess @ y - 1.0, "jac": lambda y: excess}
    ]
    if w_max < 1.0:
        # Tipe "ineq" berarti kendala pertidaksamaan, dan SciPy mensyaratkan
        # fungsinya bernilai LEBIH BESAR ATAU SAMA DENGAN NOL.
        #
        # "w_max * jumlah(y) - y >= 0" setara dengan "y_i <= w_max * jumlah(y)",
        # yang setelah normalisasi berarti bobot tidak melebihi w_max.
        #
        # np.eye(n) adalah matriks identitas: angka 1 di diagonal, 0 di lainnya.
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda y: w_max * np.sum(y) - y,
                "jac": lambda y: w_max * np.ones((n, n)) - np.eye(n),
            }
        )

    # ===== TITIK AWAL HARUS SUDAH MEMENUHI KENDALA =====
    # Kendala "excess @ y = 1" menuntut skala y yang bergantung pada BESAR
    # excess return. Kalau excess kecil (misalnya 0.013), y yang dibutuhkan
    # mencapai sekitar 77. Titik awal berskala bobot biasa (sekitar 0.33)
    # terlalu jauh, dan solver gagal konvergen.
    #
    # Inilah penyebab kesalahan "Max-Sharpe gagal konvergen" yang sempat
    # muncul pada data 730 dan 1000 hari.
    positif = excess > 0
    titik_awal = []

    # (a) seluruh bobot pada agen berexcess positif, ditimbang besar excess-nya
    dasar = np.where(positif, excess, 0.0)
    if dasar.sum() > 0:
        titik_awal.append(dasar / (excess @ (dasar / dasar.sum())) / dasar.sum())

    # (b) satu titik per agen berexcess positif.
    # np.flatnonzero(array_boolean) mengembalikan POSISI elemen bernilai True.
    #   excess  : [-0.04, 0.013, -0.03]
    #   positif : [  F  ,   T  ,   F  ]
    #   flatnonzero -> [1]
    #   y0 = [0, 1/0.013, 0] = [0, 76.9, 0]
    # Cek: excess @ y0 = 0.013 * 76.9 = 1  -> kendala langsung terpenuhi
    for i in np.flatnonzero(positif):
        y0 = np.zeros(n)
        y0[i] = 1.0 / excess[i]
        titik_awal.append(y0)

    # (c) bobot merata di antara agen berexcess positif, lalu diskalakan
    if positif.any():
        y0 = positif.astype(float) / positif.sum()
        titik_awal.append(y0 / (excess @ y0))

    # (d) beberapa titik acak yang juga diskalakan agar memenuhi kendala
    rng = np.random.default_rng(0)
    for _ in range(6):
        y0 = rng.dirichlet(np.ones(n))
        denom = excess @ y0
        if denom > 1e-9:
            titik_awal.append(y0 / denom)

    best_y, best_f = None, np.inf
    for y0 in titik_awal:
        y0 = np.clip(np.asarray(y0, dtype=float), 0.0, None)
        # np.isfinite memeriksa tidak ada NaN atau tak hingga
        if not np.isfinite(y0).all() or y0.sum() <= 0:
            continue
        res = minimize(
            fun, y0, jac=jac, method="SLSQP",
            bounds=[(0.0, None)] * n,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-10},
        )
        if res.success and res.fun < best_f:
            best_y, best_f = res.x, res.fun

    if best_y is not None:
        # _clean sekaligus menormalkan y jadi w (karena membagi dengan jumlah)
        result = portfolio_metrics(_clean(best_y), mu, cov, rf=rf)
        result["status"] = "optimal"
        return result

    # ===== MEKANISME CADANGAN =====
    # Kalau seluruh titik awal tetap gagal, sapu 220 nilai lambda pada
    # mean_variance lalu ambil Sharpe tertinggi. Lebih lambat, tetapi tidak
    # bergantung pada penskalaan variabel y sehingga jauh lebih tahan banting.
    #
    # np.logspace(-3, 6, 220) menghasilkan 220 angka dari 10^-3 sampai 10^6,
    # berjarak merata pada skala logaritmik.
    terbaik = None
    for lam in np.logspace(-3, 6, 220):
        try:
            kandidat = mean_variance(mu, cov, lam=float(lam), w_max=w_max)
        except (RuntimeError, ValueError):
            continue
        if terbaik is None or kandidat["sharpe"] > terbaik["sharpe"]:
            terbaik = kandidat

    if terbaik is None:
        raise RuntimeError("Max-Sharpe gagal konvergen.")

    terbaik["status"] = "optimal (sapuan lambda)"
    return terbaik


# =============================================================================
#  KALIBRASI LAMBDA
# =============================================================================

def implied_lambda(w: np.ndarray, cov: np.ndarray, mu: np.ndarray) -> float:
    """
    KEGUNAAN  Menghitung nilai lambda yang membuat mean_variance menghasilkan
              bobot w tertentu.

    RUMUS     lambda* = S / (2 * sigma_p)

              Diturunkan dari syarat optimum interior: pada titik optimum
              berlaku mu = 2*lambda*Sigma*w + konstanta, sehingga lambda bisa
              dibaca dari Sharpe ratio dan volatilitas.

    KOREKSI TERHADAP LAPORAN
        Laporan awal menyebut lambda* = 16.10. Angka itu sebenarnya nilai
        SHARPE RATIO, bukan lambda. Nilai lambda yang benar sekitar 108.75,
        yaitu 16.10 dibagi (2 x 0.0740).
        Sudah diuji: lambda = 16.10 justru menghasilkan solusi sudut (1,0,0),
        yaitu persis masalah yang laporan katakan terjadi pada lambda = 0.1.
    """
    from .statistics import portfolio_volatility, sharpe_ratio

    vol = portfolio_volatility(w, cov)
    if vol <= 1e-12:
        return 0.0
    return sharpe_ratio(w, mu, cov) / (2.0 * vol)


def calibrate_lambda(mu: np.ndarray, cov: np.ndarray, w_max: float = 1.0) -> float:
    """
    KEGUNAAN  Menghitung lambda dari DATA, bukan menebaknya.
              Caranya: cari dulu portofolio Sharpe-maksimum, lalu baca
              lambda yang sesuai dengannya.

    DIPAKAI OLEH  tab Analisis Sensitivitas, sebagai titik tengah sapuan.
    """
    tangency = max_sharpe(mu, cov, w_max=w_max)
    return implied_lambda(tangency["weights"], cov, mu)


# =============================================================================
#  MIN-VARIANCE (SKENARIO S4)
# =============================================================================

def min_variance(cov: np.ndarray, mu: np.ndarray, w_max: float = 1.0) -> dict:
    """
    KEGUNAAN  Mencari bobot dengan RISIKO TERENDAH.

    FUNGSI TERPENDEK TETAPI TERPENTING UNTUK HASIL AKHIR PROYEK.

    PERHATIKAN: mu TIDAK MUNCUL di fungsi tujuan. Parameter mu hanya dipakai
    untuk menghitung metrik pelaporan SETELAH bobotnya ketemu.

    KENAPA INI PENTING
        Estimasi return harapan (mu) jauh lebih sulit daripada estimasi
        kovarians (Sigma). Merton (1980) menunjukkan mu butuh periode sangat
        panjang untuk konvergen, sedangkan Sigma konvergen jauh lebih cepat.

        Ketika hasil empiris menunjukkan TIDAK ADA agen yang untung,
        memaksimalkan Sharpe jadi tidak bermakna karena pembilangnya sendiri
        negatif. Fungsi ini TETAP VALID karena tidak menyentuh mu sama sekali.
    """
    cov = np.asarray(cov, dtype=float)
    mu = np.asarray(mu, dtype=float)
    n = cov.shape[0]

    # Fungsi tujuannya hanya w'Sigma w, tanpa suku return sama sekali
    w = _minimize_simplex(
        lambda w: w @ cov @ w,          # fungsi tujuan
        lambda w: 2.0 * (cov @ w),      # turunannya
        n,
        w_max,
    )
    return portfolio_metrics(_clean(w), mu, cov)


# =============================================================================
#  FORMULA 4: CVaR
# =============================================================================

def cvar_optimize(
    returns: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    alpha: float = 0.95,
    w_max: float = 1.0,
    target_return: float | None = None,
) -> dict:
    """
    KEGUNAAN  Formula 4. Mencari bobot yang meminimalkan kerugian rata-rata
              pada EKOR distribusi (hari-hari terburuk).

    BEDANYA DENGAN MEAN-VARIANCE
        Mean-variance menimbang SELURUH sebaran return.
        CVaR hanya peduli pada sebagian kecil hari terburuk.

    ARTI ALPHA
        alpha = 0.95 -> fokus pada 5 persen hari terburuk
        alpha = 0.99 -> fokus pada 1 persen hari terburuk (lebih konservatif)

    DISELESAIKAN SEBAGAI PROGRAM LINEAR
        Berbeda dari fungsi lain di berkas ini yang kuadratik, seluruh
        kendala CVaR bersifat LINEAR, sehingga dipakai linprog dengan metode
        HiGHS, bukan SLSQP.

    SUSUNAN VARIABEL (contoh n=3, T=5, jadi n_var = 9)
        posisi : 0   1   2  |  3   |  4   5   6   7   8
        isi    : w0  w1  w2 | zeta |  u0  u1  u2  u3  u4

        w    = bobot yang dicari
        zeta = ambang. Pada optimum, nilainya SAMA DENGAN VaR
        u    = variabel bantu, menampung kelebihan kerugian di atas zeta
    """
    R = np.asarray(returns, dtype=float)
    mu = np.asarray(mu, dtype=float)

    # R.shape mengembalikan tuple (jumlah_baris, jumlah_kolom), langsung
    # dibongkar jadi dua variabel
    T, n = R.shape
    n_var = n + 1 + T

    # ===== FUNGSI TUJUAN =====
    # Vektor c berisi koefisien tiap variabel pada fungsi tujuan.
    c = np.zeros(n_var)
    c[n] = 1.0                              # koefisien zeta = 1
    c[n + 1 :] = 1.0 / ((1.0 - alpha) * T)  # koefisien tiap u
    # Bagian w diberi koefisien 0 karena tidak muncul langsung pada fungsi
    # tujuan CVaR.
    #
    # Contoh: alpha=0.95, T=450 -> koefisien u = 1/22.5 = 0.0444
    # Artinya rata-rata diambil hanya atas 5 persen hari terburuk.

    # ===== KENDALA PERTIDAKSAMAAN =====
    # Bentuk baku linprog: A_ub @ x <= b_ub
    #
    # Kendala aslinya: u_t >= -(return portofolio) - zeta
    # Dipindah ke bentuk baku: -R@w - zeta - u <= 0
    A_ub = np.zeros((T, n_var))
    A_ub[:, :n] = -R          # notasi [:, :n] = semua baris, kolom 0..n-1
    A_ub[:, n] = -1.0         # kolom zeta
    A_ub[:, n + 1 :] = -np.eye(T)  # kolom u, matriks identitas negatif
    b_ub = np.zeros(T)

    # Kendala tambahan opsional: return minimum tertentu.
    # Tidak dipakai aplikasi saat ini, tetapi disediakan untuk pengembangan.
    if target_return is not None:
        row = np.zeros(n_var)
        row[:n] = -mu
        A_ub = np.vstack([A_ub, row])
        b_ub = np.append(b_ub, -float(target_return))

    # ===== KENDALA PERSAMAAN =====
    # Jumlah bobot = 1. Hanya bagian w yang diberi koefisien 1;
    # zeta dan u tidak ikut dijumlahkan.
    A_eq = np.zeros((1, n_var))
    A_eq[0, :n] = 1.0
    b_eq = np.array([1.0])

    # ===== BATAS TIAP VARIABEL =====
    bounds = (
        [(0.0, w_max)] * n      # bobot antara 0 dan w_max
        + [(None, None)]        # zeta BEBAS, karena VaR bisa negatif
        + [(0.0, None)] * T     # u wajib tidak negatif
    )

    res = linprog(
        c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs"
    )
    if not res.success:
        raise RuntimeError(f"CVaR gagal diselesaikan: {res.message}")

    # res.x adalah vektor solusi panjang n_var. Ambil bagian bobotnya saja.
    w = _clean(res.x[:n])
    result = portfolio_metrics(w, mu, cov)

    # Nilai fungsi tujuan pada optimum = CVaR
    result["cvar"] = float(res.fun)
    # Variabel zeta pada optimum = VaR. Kesetaraan ini adalah hasil teoretis
    # dari rumusan Rockafellar dan Uryasev (2000), bukan kebetulan kode.
    result["var"] = float(res.x[n])
    result["alpha"] = alpha
    result["status"] = "optimal"
    return result


# =============================================================================
#  PEMBANTU VISUALISASI
# =============================================================================

def efficient_frontier(
    mu: np.ndarray,
    cov: np.ndarray,
    n_points: int = 40,
    w_max: float = 1.0,
) -> list[dict]:
    """
    KEGUNAAN  Menelusuri kurva efficient frontier dengan menyapu nilai lambda.
    KELUAR    list berisi dict hasil mean_variance pada tiap lambda
    """
    anchor = calibrate_lambda(mu, cov, w_max=w_max)
    anchor = anchor if anchor > 0 else 1.0
    grid = np.logspace(np.log10(anchor * 0.02), np.log10(anchor * 50.0), n_points)

    frontier = []
    for lam in grid:
        try:
            frontier.append(mean_variance(mu, cov, lam=float(lam), w_max=w_max))
        except (RuntimeError, ValueError):
            continue
    return frontier


def random_portfolios(
    mu: np.ndarray, cov: np.ndarray, n_samples: int = 4000, seed: int = 7
) -> dict:
    """
    KEGUNAAN  Membangkitkan ribuan bobot acak sebagai LATAR grafik
              efficient frontier di Tab 3. Murni visual.

    KELUAR    dict berisi weights (4000,3), return, volatility, sharpe
    """
    rng = np.random.default_rng(seed)

    # Dirichlet menghasilkan bobot yang selalu berjumlah 1
    weights = rng.dirichlet(np.ones(len(mu)), size=n_samples)

    # Matriks (4000,3) @ array (3,) -> array (4000,)
    rets = weights @ np.asarray(mu)

    # np.einsum adalah notasi Einstein untuk operasi array.
    # Pola "ij,jk,ik->i" berarti: untuk setiap baris i, hitung w_i' Sigma w_i.
    # Cara ini menghitung volatilitas 4000 portofolio SEKALIGUS tanpa
    # perulangan Python, jadi jauh lebih cepat.
    vols = np.sqrt(np.einsum("ij,jk,ik->i", weights, cov, weights))

    # errstate menekan peringatan pembagian nol; hasilnya sudah ditangani
    # oleh np.where di baris berikutnya.
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpes = np.where(vols > 0, rets / vols, 0.0)

    return {"weights": weights, "return": rets, "volatility": vols, "sharpe": sharpes}
