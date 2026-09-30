"""
===============================================================================
 datasets.py  -  MEMBACA DATA DAN MENGHITUNG STATISTIK
===============================================================================

APA ISI BERKAS INI
    Dua tugas:
      1. Membaca berkas CSV harga dari folder data/
      2. Mengubah return harian agen menjadi mu, sigma, dan rho

POSISI DALAM ALUR PROGRAM
    CSV  ->  [ load_prices ]  ->  agents.py  ->  [ from_agent_returns ]
                DataFrame          return agen        mu, sigma, rho

SUMBER DATA
    Hanya satu: berkas CSV harga nyata hasil unduhan dari Binance.
    Tidak ada data sintetis maupun data asumsi. Setiap angka pada aplikasi
    dapat ditelusuri kembali ke harga pasar.
===============================================================================
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Alamat folder data/. __file__ adalah alamat berkas ini sendiri.
#   .resolve()   -> ubah jadi alamat lengkap
#   .parent      -> naik satu tingkat (dari datasets.py ke folder core/)
#   .parent lagi -> naik lagi (dari core/ ke folder proyek)
#   / "data"     -> masuk ke folder data
# Cara ini membuat program tetap menemukan folder data walau dijalankan
# dari lokasi mana pun.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass
class AgentData:
    """
    KEGUNAAN
        Wadah yang menyatukan lima potong data agar bisa dioper antar fungsi
        sebagai SATU objek, bukan lima argumen terpisah.

    ISI
        mu      -> array (3,)   return harapan tiap agen
        sigma   -> array (3,)   volatilitas tiap agen
        rho     -> array (3,3)  matriks korelasi antar agen
        returns -> DataFrame    return harian mentah, dipakai untuk CVaR
        names   -> list         nama ketiga agen
        source  -> str          nama berkas asalnya, untuk ditampilkan

    @dataclass ITU APA
        Dekorator, yaitu perintah yang menambah kemampuan pada kelas di
        bawahnya. Tanpa ini kita harus menulis sendiri fungsi __init__ untuk
        mengisi kelima bidang tersebut; dengan dekorator ini Python
        membuatkannya otomatis.
    """

    mu: np.ndarray
    sigma: np.ndarray
    rho: np.ndarray
    returns: pd.DataFrame

    # field(default_factory=list) berarti nilai bawaannya daftar kosong.
    # Menulis "names: list = []" berbahaya di Python karena daftar itu akan
    # DIBAGI oleh semua objek yang dibuat, jadi dataclass melarangnya.
    names: list[str] = field(default_factory=list)
    source: str = ""

    @property
    def n_agents(self) -> int:
        """
        @property membuat ini bisa dipanggil seperti bidang biasa:
            data.n_agents        <- tanpa tanda kurung
        padahal sebenarnya sebuah fungsi.
        """
        return len(self.mu)


def daftar_dataset() -> dict[str, Path]:
    """
    KEGUNAAN  Mencari semua berkas CSV di folder data/, untuk mengisi menu
              pilihan di sidebar.

    KELUAR    dict {nama berkas: alamat lengkap}
              contoh: {"prices_730.csv": Path("/.../data/prices_730.csv")}
    """
    # Kalau foldernya belum ada, kembalikan dict kosong. app.py menanganinya
    # dengan menampilkan pesan "jalankan fetch_data.py dulu".
    if not DATA_DIR.exists():
        return {}

    # Bentuk {kunci: nilai for x in daftar} disebut dictionary comprehension,
    # cara ringkas menulis perulangan yang membangun dict.
    #   glob("*.csv") -> cari semua berkas berakhiran .csv
    #   sorted(...)   -> urutkan menurut abjad, supaya urutan menu tetap
    #   p.name        -> nama berkasnya saja, contoh "prices_730.csv"
    #   p             -> alamat lengkapnya
    return {p.name: p for p in sorted(DATA_DIR.glob("*.csv"))}


def load_prices(path: str | Path) -> pd.DataFrame:
    """
    KEGUNAAN  Membaca CSV harga menjadi DataFrame yang siap dipakai agen.

    MASUK     alamat berkas CSV
    KELUAR    DataFrame (T, 2) dengan indeks tanggal dan kolom BTC, ETH

    FORMAT CSV YANG DIHARAPKAN
        Tanggal,BTC,ETH
        2024-09-29,65602.01,2657.62
        2024-09-30,63327.59,2602.23
    """
    # Hasil read_csv masih berindeks angka 0,1,2,... dan kolom tanggal masih
    # berupa teks biasa.
    df = pd.read_csv(path)

    # Cari kolom mana yang berisi tanggal.
    #   next(generator, cadangan) -> ambil elemen pertama yang cocok.
    #                                Kalau tidak ada, kembalikan cadangan (None).
    #   c.lower()                 -> ubah nama kolom jadi huruf kecil, supaya
    #                                "Tanggal", "TANGGAL", "tanggal" sama saja
    #   x in (a, b, c)            -> periksa apakah x salah satu dari a, b, c
    kolom_tanggal = next(
        (c for c in df.columns if c.lower() in ("tanggal", "date", "time", "datetime")),
        None,
    )

    # raise menghentikan program dengan pesan jelas. Tanpa baris ini,
    # kesalahan baru muncul jauh di dalam program dengan pesan membingungkan.
    if kolom_tanggal is None:
        raise ValueError("Kolom tanggal tidak ditemukan pada CSV.")

    # Ubah teks tanggal jadi tipe tanggal sungguhan, supaya bisa diurutkan.
    df[kolom_tanggal] = pd.to_datetime(df[kolom_tanggal])

    # set_index  -> jadikan kolom tanggal sebagai indeks baris
    # sort_index -> urutkan dari tanggal lama ke baru
    #
    # PENGURUTAN INI PENTING. Seluruh perhitungan berikutnya memakai rolling
    # dan shift, yang bekerja menurut urutan baris. Kalau CSV tersimpan
    # terbalik, return akan jadi kebalikannya dan seluruh hasil salah TANPA
    # pesan kesalahan apa pun.
    df = df.set_index(kolom_tanggal).sort_index()
    df.index.name = "Tanggal"

    # Buang kolom non-angka secara otomatis (misalnya kolom Volume bertipe
    # teks yang ikut terbawa dari sumber data).
    harga = df.select_dtypes(include="number")
    if harga.empty:
        raise ValueError("Tidak ada kolom harga numerik pada CSV.")

    # ffill (forward fill) -> isi sel kosong dengan nilai terakhir di atasnya.
    #                         Menangani hari libur bursa.
    # dropna               -> buang baris yang MASIH kosong, biasanya baris
    #                         paling awal karena tidak ada nilai di atasnya.
    #
    # Contoh: [NaN, 10, NaN, 12] -> ffill -> [NaN, 10, 10, 12]
    #                             -> dropna -> [10, 10, 12]
    return harga.ffill().dropna()


def price_summary(prices: pd.DataFrame) -> pd.DataFrame:
    """
    KEGUNAAN  Meringkas deret harga untuk ditampilkan di Tab 1.
              MURNI TAMPILAN, tidak dipakai perhitungan lain.

    MASUK     DataFrame (T, 2)
    KELUAR    DataFrame (2, 5) - satu baris per aset
    """
    # pct_change() menghitung (P[t] - P[t-1]) / P[t-1] untuk setiap baris.
    # Baris pertama jadi NaN, lalu dibuang oleh dropna.
    # Dikali 100 supaya jadi persen.
    rets = prices.pct_change().dropna() * 100.0

    return pd.DataFrame(
        {
            # iloc[0] = baris pertama menurut posisi, yaitu hari paling awal
            "Harga awal": prices.iloc[0].round(2),
            # iloc[-1] = baris terakhir. Angka negatif menghitung dari belakang.
            "Harga akhir": prices.iloc[-1].round(2),
            "Perubahan (%)": ((prices.iloc[-1] / prices.iloc[0] - 1) * 100).round(2),
            "Return harian rata-rata (%)": rets.mean().round(4),
            # ddof=1 berarti pembagi n-1, bukan n. Itu estimator tak bias
            # untuk sampel. Kalau diubah jadi ddof=0, sigma sedikit lebih kecil.
            "Volatilitas harian (%)": rets.std(ddof=1).round(4),
        }
    )


def from_agent_returns(returns: pd.DataFrame, source: str = "") -> AgentData:
    """
    KEGUNAAN  ===== FUNGSI PALING PENTING DI BERKAS INI =====
              Mengubah return harian agen menjadi mu, sigma, dan rho.

    MASUK     DataFrame (T, 3) - return harian tiap agen
    KELUAR    objek AgentData

    PERUBAHAN BENTUK DATA
        df        : (450, 3)  tabel besar
        df.mean() : (3,)      satu nilai per kolom
        df.std()  : (3,)      satu nilai per kolom
        df.corr() : (3, 3)    matriks korelasi

    INI SATU-SATUNYA TEMPAT mu, sigma, DAN rho DIHITUNG.
    Seluruh angka di Tab 2, 3, 4, dan 5 berasal dari tiga baris di bawah.
    """
    df = returns.dropna()

    # shape[0] = jumlah baris. Dengan kurang dari 3 baris, simpangan baku dan
    # korelasi tidak bermakna, jadi berhenti lebih awal dengan pesan jelas.
    if df.shape[0] < 3:
        raise ValueError("Return agen terlalu pendek untuk estimasi statistik.")

    return AgentData(
        # mean() pada DataFrame menghitung rata-rata TIAP KOLOM, bukan seluruh
        # tabel. Hasilnya Series panjang 3.
        # to_numpy() mengubahnya jadi array numpy, karena optimizer.py bekerja
        # dengan numpy, bukan pandas.
        mu=df.mean().to_numpy(),
        sigma=df.std(ddof=1).to_numpy(),
        # corr() menghitung korelasi antar kolom, hasilnya DataFrame (3,3)
        rho=df.corr().to_numpy(),
        # returns disimpan utuh karena cvar_optimize butuh data harian mentah,
        # bukan sekadar ringkasannya
        returns=df,
        names=list(df.columns),
        source=source,
    )


def summary_table(data: AgentData) -> pd.DataFrame:
    """
    KEGUNAAN  Menyusun tabel statistik agen untuk ditampilkan di Tab 2.
              MURNI TAMPILAN. Kalau fungsi ini dihapus, perhitungan program
              tetap benar dan hanya tabelnya yang hilang.

    MASUK     objek AgentData
    KELUAR    DataFrame (3, 3)
    """
    # KENAPA TIDAK DITULIS data.mu / data.sigma SAJA?
    # Karena kalau ada agen yang sigma-nya nol (misalnya agen yang tidak
    # pernah mengambil posisi sehingga return-nya nol terus), pembagian biasa
    # menghasilkan pembagian dengan nol. Python memunculkan peringatan dan
    # mengisi hasilnya dengan inf atau nan, yang merusak seluruh tabel.
    #
    # np.divide dengan dua argumen tambahan mencegah itu:
    #   out=np.zeros_like(data.mu) -> wadah hasil, diisi nol lebih dulu.
    #                                 zeros_like membuat array nol dengan
    #                                 bentuk dan tipe SAMA PERSIS seperti mu.
    #   where=data.sigma > 0       -> array True/False. Pembagian HANYA
    #                                 dikerjakan di posisi True. Posisi False
    #                                 dilewati dan nilainya tetap seperti
    #                                 isi out, yaitu nol.
    #
    # Contoh bila agen 3 volatilitasnya nol:
    #   mu    = [2.0, 1.0, 0.5]
    #   sigma = [4.0, 0.0, 2.0]
    #   where = [  T,   F,   T]
    #   hasil = [0.5, 0.0, 0.25]     <- posisi kedua tidak dibagi
    sharpe = np.divide(
        data.mu, data.sigma, out=np.zeros_like(data.mu), where=data.sigma > 0
    )

    return pd.DataFrame(
        # Argumen pertama berupa dict:
        #   teks sebelum titik dua -> nama kolom
        #   array setelah titik dua -> isi kolom
        # Ketiga array harus sama panjang, yaitu 3.
        {
            # np.round(arr, 4) membulatkan setiap elemen ke 4 desimal.
            # Ini HANYA mengubah angka yang tertulis di layar. Perhitungan di
            # belakangnya tetap memakai nilai penuh.
            "Return harapan (mu)": np.round(data.mu, 4),
            "Volatilitas (sigma)": np.round(data.sigma, 4),
            "Sharpe (mu/sigma)": np.round(sharpe, 3),
        },
        # index menetapkan LABEL BARIS. Tanpa ini, pandas memberi nomor
        # 0, 1, 2 dan pembaca tidak tahu baris mana milik agen mana.
        index=data.names,
    )
