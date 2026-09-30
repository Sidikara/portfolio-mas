"""
===============================================================================
 scenarios.py  -  PERBANDINGAN EMPAT SKENARIO ALOKASI
===============================================================================

APA ISI BERKAS INI
    Menyusun beberapa cara membagi modal, menghitung metriknya, lalu
    menyajikannya sebagai tabel perbandingan di Tab 2.

POSISI DALAM ALUR PROGRAM
    optimizer.py  ->  [ scenarios.py ]  ->  tabel di Tab 2
     cari bobot        bandingkan

EMPAT SKENARIO
    S1 Equal-weight   1/N untuk semua           tidak perlu mu
    S2 Naive          bobot tetap dari laporan  tidak perlu mu
    S3 Max-Sharpe     return per risiko         PERLU mu
    S4 Min-Variance   risiko terendah           tidak perlu mu

    Ditambah: referensi alokasi 100 persen pada satu agen, dan benchmark
    buy-and-hold aset dasar sebagai pembanding jujur.

KENAPA S4 ADA
    Estimasi return harapan (mu) jauh lebih sulit daripada estimasi kovarians
    (Sigma). Merton (1980) menunjukkan mu butuh periode sangat panjang untuk
    konvergen, sedangkan Sigma konvergen jauh lebih cepat.

    Ketika tidak ada agen yang untung, memaksimalkan Sharpe jadi tidak
    bermakna karena pembilangnya sendiri negatif. S4 tetap bermakna dalam
    kondisi itu karena hanya bersandar pada Sigma.
===============================================================================
"""

import numpy as np
import pandas as pd

from .optimizer import max_sharpe, min_variance
from .risk import herfindahl
from .statistics import portfolio_metrics


# =============================================================================
#  TIGA FUNGSI BOBOT SEDERHANA (tanpa optimisasi)
# =============================================================================

def equal_weight(n: int) -> np.ndarray:
    """Bobot merata: 1/n untuk setiap agen. n=3 -> [0.333, 0.333, 0.333]"""
    return np.full(n, 1.0 / n)


def naive_weight(n: int) -> np.ndarray:
    """
    Bobot tetap dari laporan: 50 persen, 30 persen, 20 persen.
    Kalau jumlah agen bukan 3, turun ke alokasi merata supaya panjang
    arraynya tetap cocok.
    """
    if n == 3:
        return np.array([0.50, 0.30, 0.20])
    return equal_weight(n)


def single_agent_weight(n: int, index: int) -> np.ndarray:
    """
    Seluruh modal ke satu agen. Dipakai sebagai REFERENSI pembanding.
    n=3, index=1 -> [0.0, 1.0, 0.0]
    """
    w = np.zeros(n)
    w[index] = 1.0
    return w


# =============================================================================
#  DUA FUNGSI PEMBANTU
# =============================================================================

def _drawdown(series_pct: np.ndarray) -> float:
    """
    Penurunan terdalam dari puncak kurva ekuitas, dalam persen.
    Sama seperti max_drawdown di risk.py, tetapi menerima deret return
    portofolio yang SUDAH JADI, bukan matriks return agen beserta bobot.
    """
    equity = np.cumprod(1.0 + np.asarray(series_pct, dtype=float) / 100.0)
    peak = np.maximum.accumulate(equity)
    return float(np.max((peak - equity) / peak) * 100.0)


def _total_return(series_pct: np.ndarray) -> float:
    """
    Total return sepanjang periode, dalam persen.
    Nilai ekuitas terakhir dikurangi 1, dikali 100.
      equity[-1] = 1.5  ->  (1.5 - 1) * 100 = 50 persen
    """
    equity = np.cumprod(1.0 + np.asarray(series_pct, dtype=float) / 100.0)
    return float((equity[-1] - 1.0) * 100.0)


# =============================================================================
#  MEMBANGUN SELURUH SKENARIO
# =============================================================================

def build_scenarios(
    mu: np.ndarray,
    cov: np.ndarray,
    w_max: float = 1.0,
    include_single: bool = True,
    agent_names: list[str] | None = None,
    returns: pd.DataFrame | None = None,
    benchmark: pd.Series | None = None,
    benchmark_label: str = "Benchmark: Buy & Hold BTC",
) -> list[dict]:
    """
    KEGUNAAN  Membangun seluruh skenario beserta metriknya.
              INI FUNGSI UTAMA berkas ini.

    MASUK     mu (3,), cov (3,3), w_max
              returns   = DataFrame return harian (opsional). Kalau ada,
                          drawdown dan total return ikut dihitung.
              benchmark = Series return aset dasar (opsional)

    KELUAR    list berisi dict. Setiap dict satu skenario.

    STRUKTUR HASIL
        rows = [
          {"weights": [...], "return": 0.21, "label": "S1 ...", "hhi": 0.333, ...},
          {"weights": [...], "return": 0.19, "label": "S2 ...", "hhi": 0.380, ...},
          ...
        ]
    """
    n = len(mu)
    # "a or b" berarti pakai a kalau a bernilai benar, selain itu b
    names = agent_names or [f"Agen {i + 1}" for i in range(n)]
    rows: list[dict] = []

    # ===== S1 DAN S2 =====
    # Perulangan atas list berisi TUPLE. Setiap tuple dibongkar jadi dua
    # variabel sekaligus (label dan w). Teknik ini disebut TUPLE UNPACKING.
    for label, w in [
        ("S1 Equal-weight", equal_weight(n)),
        ("S2 Naive", naive_weight(n)),
    ]:
        m = portfolio_metrics(w, mu, cov)
        # portfolio_metrics mengembalikan dict. Dict bersifat MUTABLE, jadi
        # kita bisa menambahkan kunci baru setelahnya.
        m["label"] = label
        # Kunci "kelompok" dipakai belakangan untuk membedakan mana portofolio,
        # mana agen tunggal, mana benchmark, supaya perbandingan tidak tercampur
        m["kelompok"] = "portofolio"
        rows.append(m)

    # ===== S3: PENJAGAAN PALING PENTING DI BERKAS INI =====
    # np.any(kondisi) bernilai True kalau SETIDAKNYA SATU elemen memenuhi.
    # (Berbeda dari np.all yang mensyaratkan seluruhnya.)
    #
    # Kalau tidak ada agen dengan return harapan positif, S3 DIHILANGKAN
    # karena memaksimalkan Sharpe jadi tidak bermakna: pembilangnya sendiri
    # negatif, sehingga portofolio ber-Sharpe tertinggi justru yang paling
    # berisiko.
    #
    # Tanpa baris if ini, aplikasi akan berhenti dengan layar merah berisi
    # traceback, bukan peringatan kuning yang rapi.
    if np.any(mu > 0):
        m = max_sharpe(mu, cov, w_max=w_max)
        m["label"] = "S3 Max-Sharpe"
        m["kelompok"] = "portofolio"
        rows.append(m)

    # ===== S4: DIBANGUN TANPA SYARAT APA PUN =====
    # min_variance sama sekali tidak memakai mu, jadi selalu bisa dihitung.
    m = min_variance(cov, mu, w_max=w_max)
    m["label"] = "S4 Min-Variance"
    m["kelompok"] = "portofolio"
    rows.append(m)

    # ===== REFERENSI: SATU AGEN SAJA =====
    if include_single:
        for i in range(n):
            m = portfolio_metrics(single_agent_weight(n, i), mu, cov)
            m["label"] = f"Referensi: {names[i]} saja"
            m["kelompok"] = "agen tunggal"
            rows.append(m)

    # ===== TAMBAHKAN METRIK KE SETIAP SKENARIO =====
    # Perulangan ini menambah kunci baru ke setiap dict yang SUDAH ADA di list.
    for row in rows:
        row["hhi"] = herfindahl(row["weights"])
        # "returns is not None" berarti data return harian tersedia
        if returns is not None:
            # Matriks (T,3) @ array (3,) -> array (T,) return portofolio harian
            seri = returns.to_numpy() @ row["weights"]
            row["drawdown"] = _drawdown(seri)
            row["total_return"] = _total_return(seri)

    # ===== BENCHMARK BUY-AND-HOLD =====
    # Pembanding jujur: apa jadinya kalau modal hanya disimpan di aset dasar
    # tanpa strategi apa pun. Tanpa ini, tidak ada konteks untuk menilai
    # apakah strategi aktif benar-benar berguna.
    if benchmark is not None:
        b = benchmark.dropna().to_numpy()
        rows.append(
            {
                "label": benchmark_label,
                "kelompok": "benchmark",
                # Benchmark bukan portofolio agen, jadi bobotnya diisi NaN
                "weights": np.full(n, np.nan),
                "return": float(np.mean(b)),
                "volatility": float(np.std(b, ddof=1)),
                "sharpe": float(np.mean(b) / np.std(b, ddof=1))
                if np.std(b, ddof=1) > 1e-12
                else 0.0,
                "hhi": np.nan,
                "drawdown": _drawdown(b),
                "total_return": _total_return(b),
            }
        )

    return rows


def scenarios_dataframe(rows: list[dict], decimals: int = 4) -> pd.DataFrame:
    """
    KEGUNAAN  Mengubah list berisi dict menjadi tabel siap tampil.
              MURNI TAMPILAN.

    MASUK     list dict hasil build_scenarios
    KELUAR    DataFrame siap ditampilkan Streamlit
    """
    # any(...) bernilai True kalau setidaknya satu elemen memenuhi syarat.
    # Dipakai untuk memutuskan apakah kolom drawdown perlu ditampilkan.
    ada_dd = any("drawdown" in r for r in rows)
    records = []

    for row in rows:
        w = row["weights"]

        # Baris benchmark bobotnya NaN, ditampilkan sebagai "n/a".
        # join menyambung beberapa teks dengan pemisah.
        # f"{x:.3f}" membulatkan ke 3 desimal, hasilnya "(0.263, 0.539, 0.198)"
        bobot = (
            "n/a"
            if np.all(np.isnan(w))
            else "(" + ", ".join(f"{x:.3f}" for x in w) + ")"
        )

        rec = {
            "Skenario": row["label"],
            "Bobot (w)": bobot,
            "Return (%/hari)": round(row["return"], 4),
            "Volatilitas (%/hari)": round(row["volatility"], decimals),
            "Sharpe": round(row["sharpe"], 3),
            # row.get(kunci, cadangan) mengambil nilai kalau kuncinya ada,
            # kalau tidak ada kembalikan cadangan. Lebih aman dari row[kunci]
            # yang akan error kalau kuncinya tidak ada.
            "HHI": round(row["hhi"], 3) if not np.isnan(row.get("hhi", np.nan)) else np.nan,
        }
        if ada_dd:
            rec["Max drawdown (%)"] = round(row.get("drawdown", np.nan), 2)
            rec["Total return (%)"] = round(row.get("total_return", np.nan), 2)
        records.append(rec)

    return pd.DataFrame(records)


# =============================================================================
#  DUA CARA MENILAI MANFAAT KOORDINASI
# =============================================================================

def coordination_benefit(rows: list[dict], mu: np.ndarray) -> dict:
    """
    KEGUNAAN  Menilai manfaat koordinasi pada dimensi RETURN.
              Membandingkan S3 dengan agen tunggal ber-Sharpe tertinggi.

    KELUAR    dict berisi perbandingan, ATAU dict KOSONG kalau tidak berlaku

    KAPAN MENGEMBALIKAN DICT KOSONG
        1. S3 tidak ada (karena tidak ada agen ber-return positif)
        2. Agen tunggal terbaik pun Sharpe-nya tidak positif

        Dalam kedua kasus itu, klaim "mengungguli agen terbaik" jadi tidak
        bermakna. app.py menanganinya dengan tidak menampilkan blok metrik ini.
    """
    # next(generator, cadangan) mengambil elemen pertama yang cocok.
    # Argumen kedua (None) adalah cadangan kalau tidak ditemukan, supaya
    # program tidak berhenti dengan kesalahan.
    optimized = next((r for r in rows if r["label"].startswith("S3")), None)

    # [r for r in rows if kondisi] disebut LIST COMPREHENSION: bangun list
    # baru berisi elemen yang lolos saringan.
    singles = [r for r in rows if r["kelompok"] == "agen tunggal"]

    # "not singles" bernilai True kalau listnya kosong
    if optimized is None or not singles:
        return {}

    # max(list, key=fungsi) mencari elemen dengan nilai FUNGSI terbesar,
    # bukan elemen terbesar itu sendiri.
    # Tanpa argumen key, Python akan mencoba membandingkan dua dict secara
    # langsung dan menghasilkan kesalahan.
    #
    # Dipilih Sharpe tertinggi, BUKAN return tertinggi, karena Sharpe adalah
    # ukuran yang adil (sudah memperhitungkan risiko).
    best = max(singles, key=lambda r: r["sharpe"])
    if best["sharpe"] <= 0:
        return {}

    return {
        "benchmark": best["label"],
        "sharpe_optimized": optimized["sharpe"],
        "sharpe_benchmark": best["sharpe"],
        # Rumus perubahan persen: (baru / lama - 1) * 100
        "sharpe_gain_pct": (optimized["sharpe"] / best["sharpe"] - 1.0) * 100.0,
        "vol_optimized": optimized["volatility"],
        "vol_benchmark": best["volatility"],
        "vol_change_pct": (optimized["volatility"] / best["volatility"] - 1.0) * 100.0,
        "return_change_pct": (optimized["return"] / best["return"] - 1.0) * 100.0
        if abs(best["return"]) > 1e-12
        else np.nan,
        "hhi_optimized": optimized["hhi"],
    }


def risk_benefit(rows: list[dict], skenario: str = "S4") -> dict:
    """
    KEGUNAAN  Menilai manfaat koordinasi pada dimensi RISIKO.

    KENAPA FUNGSI INI PENTING
        Ukuran ini TETAP BERMAKNA ketika tidak ada agen yang untung, karena
        sama sekali tidak bersandar pada estimasi return. Inilah klaim yang
        masih bisa dipertahankan pada data nyata.

    PEMBANDINGNYA
        Sengaja dipilih agen tunggal dengan volatilitas TERENDAH, bukan yang
        paling mudah dikalahkan. Memilih pembanding termudah akan membuat
        klaim terlihat bagus tetapi tidak jujur.

    KELUAR  dict berisi perbandingan volatilitas dan drawdown
    """
    portofolio = next((r for r in rows if r["label"].startswith(skenario)), None)
    singles = [r for r in rows if r["kelompok"] == "agen tunggal"]

    if portofolio is None or not singles:
        return {}

    # min(list, key=fungsi) kebalikan dari max
    teraman = min(singles, key=lambda r: r["volatility"])

    hasil = {
        "skenario": portofolio["label"],
        "pembanding": teraman["label"],
        "vol_portofolio": portofolio["volatility"],
        "vol_pembanding": teraman["volatility"],
        # Hasil NEGATIF berarti volatilitas portofolio LEBIH RENDAH, yaitu bagus
        "vol_change_pct": (portofolio["volatility"] / teraman["volatility"] - 1.0) * 100.0,
        "vol_rata_agen": float(np.mean([s["volatility"] for s in singles])),
        "hhi": portofolio["hhi"],
    }
    hasil["vol_vs_rata_pct"] = (
        portofolio["volatility"] / hasil["vol_rata_agen"] - 1.0
    ) * 100.0

    # Bagian drawdown hanya dihitung kalau datanya tersedia.
    # all(...) bernilai True kalau SELURUH elemen memenuhi syarat.
    if "drawdown" in portofolio and all("drawdown" in s for s in singles):
        dd_teraman = min(singles, key=lambda r: r["drawdown"])
        dd_rata = float(np.mean([s["drawdown"] for s in singles]))
        hasil.update(
            {
                "dd_portofolio": portofolio["drawdown"],
                "dd_pembanding": dd_teraman["drawdown"],
                "dd_label_pembanding": dd_teraman["label"],
                # Penjagaan pembagian nol: kalau drawdown pembanding nol,
                # hasilnya diisi NaN dan app.py tidak menampilkannya
                "dd_change_pct": (portofolio["drawdown"] / dd_teraman["drawdown"] - 1.0)
                * 100.0
                if dd_teraman["drawdown"] > 1e-9
                else np.nan,
                "dd_rata_agen": dd_rata,
                "dd_vs_rata_pct": (portofolio["drawdown"] / dd_rata - 1.0) * 100.0
                if dd_rata > 1e-9
                else np.nan,
            }
        )

    return hasil
