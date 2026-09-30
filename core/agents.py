"""
===============================================================================
 agents.py  -  TIER 1: TIGA AGEN STRATEGI
===============================================================================

APA ISI BERKAS INI
    Mengubah deret harga menjadi keputusan beli dan jual, lalu menghitung
    untung rugi harian dari keputusan itu. Inilah bagian yang membuat sistem
    ini disebut multi-agen.

POSISI DALAM ALUR PROGRAM
    datasets.py  ->  [ agents.py ]  ->  datasets.py  ->  optimizer.py
     baca harga      hasilkan sinyal    hitung statistik   cari bobot

TIGA AGEN
    Agen 1  Trend-Following : ikut arah tren      (SMA-50 vs SMA-200)
    Agen 2  Mean-Reversion  : lawan arah          (z-score harga)
    Agen 3  Market-Neutral  : perdagangkan selisih (spread BTC-ETH)

POLA YANG SAMA PADA KETIGA AGEN
    Langkah 1  hitung indikator dari harga
    Langkah 2  ubah indikator jadi posisi: -1 (jual), 0 (diam), +1 (beli)
    Langkah 3  posisi KEMARIN dikali return HARI INI, lalu kurangi biaya

    Sekali paham pola ini, ketiga agen langsung terbaca.

ATURAN PALING PENTING
    position.shift(1) menggeser posisi maju satu hari, sehingga posisi yang
    diputuskan kemarin menentukan untung rugi hari ini. Tanpa itu, agen
    seolah tahu harga hari ini sebelum mengambil posisi. Istilahnya
    look-ahead bias, dan seluruh hasil backtest jadi tidak sah.
===============================================================================
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Nama ketiga agen. Dipakai sebagai nama kolom pada DataFrame hasil.
AGENT_LABELS = [
    "Agen 1 - Trend-Following",
    "Agen 2 - Mean-Reversion",
    "Agen 3 - Market-Neutral",
]


@dataclass
class AgentConfig:
    """
    KEGUNAAN
        Mengumpulkan SELURUH parameter strategi di satu tempat, sehingga bisa
        diubah dari sidebar aplikasi tanpa menyentuh kode.

    ANGKA SETELAH TANDA SAMA DENGAN adalah nilai bawaan, dipakai bila tidak
    diisi saat memanggil.

    DAMPAK BILA DIUBAH (sudah diuji pada data 730 hari)
        cost_bps 5 -> 0    : semua Sharpe naik sedikit, agen terlihat lebih untung
        cost_bps 5 -> 50   : semua Sharpe turun, agen paling aktif paling terpukul
        sma 50/200 -> 10/30: agen 1 bisa berubah dari rugi jadi untung
        mr_entry 1.0 -> 2.5: agen 2 hampir tidak pernah bertransaksi (aktif 71% -> 2%)
        allow_short -> False: hari aktif agen 1 dan 2 turun drastis
        pair_beta 250 -> 30: pemanasan memendek, JUMLAH HARI EFEKTIF BERTAMBAH,
                             sehingga Sharpe SEMUA agen berubah
    """

    # --- Agen 1: Trend-Following ---
    sma_fast: int = 50    # jendela rata-rata bergerak cepat
    sma_slow: int = 200   # jendela rata-rata bergerak lambat

    # --- Agen 2: Mean-Reversion ---
    mr_window: int = 20   # jendela hitung rata-rata dan simpangan baku
    mr_entry: float = 1.0 # ambang z-score untuk mulai mengambil posisi

    # --- Agen 3: Market-Neutral ---
    pair_window: int = 30       # jendela sinyal, perlu responsif
    pair_entry: float = 1.0     # ambang z-score spread
    pair_beta_window: int = 250 # jendela estimasi hedge ratio, perlu stabil

    # --- Berlaku untuk semua agen ---
    allow_short: bool = True  # boleh ambil posisi jual (short) atau tidak
    cost_bps: float = 5.0     # biaya per perubahan posisi, satuan basis poin
                              # 1 bps = 0.01 persen, jadi 5 bps = 0.05 persen

    @classmethod
    def for_history(cls, n_days: int, **override) -> "AgentConfig":
        """
        KEGUNAAN
            Memperkecil jendela indikator kalau datanya terlalu pendek.

        MASALAH YANG DISELESAIKAN
            Parameter bawaan butuh 280 hari sebelum agen menghasilkan sinyal
            pertama (250 untuk beta + 30 untuk sinyal). Pada data 90 hari,
            SELURUH indikator akan bernilai NaN dan ketiga agen tidak pernah
            mengambil posisi sama sekali.

        ATURANNYA
            Indikator hanya boleh memakan SEPARUH panjang data, supaya separuh
            sisanya tersedia untuk menghitung mu, sigma, dan rho.

        CONTOH pada data 90 hari:
            anggaran   = 45
            sma_slow   = 45,  sma_fast = 11
            mr_window  = 15
            pair_beta  = 33,  pair_window = 12

        @classmethod ITU APA
            Fungsi yang dipanggil dari KELASNYA, bukan dari objeknya:
                AgentConfig.for_history(90)
            Parameter cls merujuk pada kelas itu sendiri, sehingga baris
            terakhir bisa menulis cls(...) untuk membuat objek baru.

        **override ITU APA
            Menerima argumen tambahan sebanyak apa pun sebagai dict, sehingga
            pemanggil bisa menimpa hasil hitungan otomatis.
        """
        # Separuh panjang data, minimal 20 supaya tidak terlalu kecil
        anggaran = max(int(n_days * 0.5), 20)

        # Pola max(min(a, b), c) membatasi nilai di antara dua sisi:
        #   min(200, anggaran) -> jangan lebih dari 200
        #   max(..., 10)       -> jangan kurang dari 10
        sma_slow = max(min(200, anggaran), 10)

        # Operator // adalah pembagian BULAT (buang angka di belakang koma).
        # sma_fast selalu seperempat sma_slow, mengikuti perbandingan 50:200.
        sma_fast = max(sma_slow // 4, 3)

        mr_window = max(min(20, anggaran // 3), 5)

        # Jendela beta dan jendela sinyal dipakai BERURUTAN, jadi JUMLAH
        # keduanya yang harus muat dalam anggaran. Beta dapat tiga perempat
        # bagian karena estimasi hedge ratio butuh lebih banyak data.
        pair_beta = max(min(250, int(anggaran * 0.75)), 20)
        pair_window = max(min(30, anggaran - pair_beta), 5)

        dasar = dict(
            sma_fast=sma_fast,
            sma_slow=sma_slow,
            mr_window=mr_window,
            pair_window=pair_window,
            pair_beta_window=pair_beta,
        )
        dasar.update(override)  # timpa dengan nilai yang diberikan pemanggil
        return cls(**dasar)

    def requires_days(self) -> int:
        """
        KEGUNAAN  Menghitung berapa hari minimum sebelum ketiga agen mulai
                  menghasilkan sinyal.
        KELUAR    satu angka. Pada nilai bawaan hasilnya 280 (= 250 + 30).
        """
        return max(self.sma_slow, self.mr_window, self.pair_beta_window + self.pair_window)


# =============================================================================
#  DUA FUNGSI PEMBANTU
#  Garis bawah di depan nama menandakan fungsi internal, hanya dipakai di
#  dalam berkas ini, tidak dipanggil dari luar.
# =============================================================================

def _to_returns(prices: pd.Series) -> pd.Series:
    """
    KEGUNAAN  Mengubah deret harga menjadi return harian dalam persen.

    MASUK     Series panjang T berisi harga
    KELUAR    Series panjang T berisi persentase

    CONTOH    masuk  : [100, 110, 99, 99]
              hitung : [NaN, (110-100)/100, (99-110)/110, (99-99)/99]
              keluar : [NaN, 10.0, -10.0, 0.0]

    Panjang dan indeksnya dipertahankan. Elemen pertama SELALU NaN karena
    tidak ada harga sebelumnya untuk dibandingkan.
    """
    return prices.pct_change() * 100.0


def _apply_costs(position: pd.Series, cost_bps: float) -> pd.Series:
    """
    KEGUNAAN  Menghitung biaya transaksi. Biaya hanya dikenakan saat posisi
              BERUBAH, bukan setiap hari.

    MASUK     Series posisi panjang T, dan satu angka biaya
    KELUAR    Series biaya panjang T

    CONTOH dengan cost_bps = 5:
        position: [ 0,  1,  1, -1,  0]
        diff()  : [NaN, 1,  0, -2,  1]     selisih dengan baris sebelumnya
        abs()   : [NaN, 1,  0,  2,  1]     buang tanda negatif
        fillna  : [  0, 1,  0,  2,  1]     NaN diganti abs(position[0]) = 0
        x 0.05  : [0.00, 0.05, 0.00, 0.10, 0.05]

    PERHATIKAN posisi keempat bernilai 2, bukan 1. Selisih dari +1 ke -1
    adalah -2, karena harus menutup beli lalu membuka jual: dua transaksi.
    """
    # diff() menghitung x[i] - x[i-1]. Baris pertama jadi NaN.
    # fillna(position.abs()) menangani baris pertama: kalau hari pertama sudah
    # berposisi, biayanya tetap dikenakan sebesar posisi itu.
    turnover = position.diff().abs().fillna(position.abs())

    # Bagi 100 untuk mengubah basis poin jadi persen (1 bps = 0.01 persen)
    return turnover * (cost_bps / 100.0)


# =============================================================================
#  AGEN 1: TREND-FOLLOWING
# =============================================================================

def trend_following_agent(
    prices: pd.DataFrame, cfg: AgentConfig, asset: str = "BTC"
) -> pd.DataFrame:
    """
    KEGUNAAN  Agen yang MENGIKUTI arah tren.
              Beli saat rata-rata cepat di atas rata-rata lambat.

    MASUK     DataFrame harga (T, 2), objek AgentConfig
    KELUAR    DataFrame (T, 4) kolom: sinyal, return, sma_cepat, sma_lambat

    LOGIKA    SMA cepat di atas SMA lambat  -> harga sedang naik  -> beli (+1)
              SMA cepat di bawah SMA lambat -> harga sedang turun -> jual (-1)
    """
    # Ambil satu kolom dari DataFrame. Hasilnya Series, bukan DataFrame.
    px = prices[asset]

    # rolling(50).mean() menghitung rata-rata 50 baris terakhir untuk setiap
    # posisi. 49 baris pertama jadi NaN karena datanya belum cukup.
    fast = px.rolling(cfg.sma_fast).mean()
    slow = px.rolling(cfg.sma_slow).mean()

    # np.where(kondisi, nilai_bila_benar, nilai_bila_salah)
    # Bekerja untuk SELURUH baris sekaligus, tanpa perulangan.
    #
    # Bagian "-1.0 if cfg.allow_short else 0.0" adalah ekspresi kondisional:
    # kalau short diizinkan, nilai sebaliknya -1; kalau tidak, 0 (tanpa posisi).
    raw = np.where(fast > slow, 1.0, -1.0 if cfg.allow_short else 0.0)

    # pd.Series(...) membungkus array numpy jadi Series dengan indeks tanggal
    # yang sama seperti harga, supaya bisa disejajarkan nanti.
    #
    # .where(kondisi, 0.0) milik pandas bekerja TERBALIK dari np.where:
    #   pertahankan nilai bila kondisi BENAR, ganti jadi 0.0 bila SALAH.
    # Di sini gunanya memaksa posisi jadi nol selama SMA lambat masih NaN.
    #
    # Kenapa perlu? Karena perbandingan "fast > slow" dengan NaN selalu
    # menghasilkan False, sehingga baris awal akan bernilai -1 padahal
    # seharusnya belum ada posisi sama sekali.
    position = pd.Series(raw, index=px.index).where(slow.notna(), 0.0)

    asset_ret = _to_returns(px)

    # ===== BARIS PALING KRUSIAL DI SELURUH BERKAS INI =====
    # shift(1) menggeser posisi maju satu hari, sehingga posisi yang
    # diputuskan KEMARIN menentukan untung rugi HARI INI.
    #
    # Tanpa shift(1), hasilnya jadi berlebihan ke arah mana pun. Sudah diuji:
    #   strategi untung : +515%  ->  +1934%
    #   strategi rugi   :  -85%  ->   -95.5%
    # Penyebabnya, posisi hari ini ditentukan memakai harga hari ini, lalu
    # dinilai memakai return hari ini juga: sebagian jadi ramalan yang
    # memenuhi dirinya sendiri.
    gross = position.shift(1).fillna(0.0) * asset_ret

    net = gross - _apply_costs(position, cfg.cost_bps)

    return pd.DataFrame(
        {
            # Dua kolom ini dipakai perhitungan:
            "sinyal": position,
            "return": net.fillna(0.0),
            # Dua kolom ini HANYA untuk menggambar grafik di Tab 1:
            "sma_cepat": fast,
            "sma_lambat": slow,
        }
    )


# =============================================================================
#  AGEN 2: MEAN-REVERSION
# =============================================================================

def mean_reversion_agent(
    prices: pd.DataFrame, cfg: AgentConfig, asset: str = "BTC"
) -> pd.DataFrame:
    """
    KEGUNAAN  Agen yang MELAWAN arah. Beli saat harga dianggap terlalu murah,
              jual saat dianggap terlalu mahal.

    MASUK     DataFrame harga (T, 2), objek AgentConfig
    KELUAR    DataFrame (T, 3) kolom: sinyal, return, zscore

    LOGIKA    z = (harga - rata-rata) / simpangan baku
              z sangat rendah  -> terlalu murah -> beli (+1)
              z sangat tinggi  -> terlalu mahal -> jual (-1)

    PERHATIKAN ARAH TANDANYA. z rendah menghasilkan +1, BUKAN -1. Strategi
    ini memang sengaja melawan arah. Kalau tandanya dibalik, agen ini berubah
    jadi trend-following dan tidak sesuai lagi dengan deskripsi di laporan.
    (Sudah diuji: Sharpe berubah dari -0.29 jadi +0.27.)
    """
    px = prices[asset]
    ma = px.rolling(cfg.mr_window).mean()
    sd = px.rolling(cfg.mr_window).std(ddof=1)

    # sd.replace(0.0, np.nan) mengganti nilai nol jadi NaN SEBELUM pembagian.
    # Kalau harga tidak bergerak sama sekali dalam jendela itu, sd bernilai
    # nol dan pembagian akan gagal. Dengan diganti NaN, hasilnya juga NaN
    # dan ditangani belakangan oleh .where(z.notna(), 0.0).
    z = (px - ma) / sd.replace(0.0, np.nan)

    # Mulai dengan Series berisi nol untuk seluruh hari
    position = pd.Series(0.0, index=px.index)

    # Bentuk position[kondisi] = nilai disebut BOOLEAN INDEXING.
    # Hanya baris yang kondisinya True yang diisi; sisanya tidak disentuh.
    #
    # Contoh:
    #   z              : [NaN, 2.5, -1.8, 0.3]
    #   z < -1.0       : [ F ,  F ,   T  ,  F ]  -> isi 1.0 di posisi 2
    #   z > 1.0        : [ F ,  T ,   F  ,  F ]  -> isi -1.0 di posisi 1
    #   position akhir : [  0,  -1,    1 ,   0 ]
    position[z < -cfg.mr_entry] = 1.0
    if cfg.allow_short:
        position[z > cfg.mr_entry] = -1.0

    # Paksa nol pada baris di mana z masih NaN (periode pemanasan)
    position = position.where(z.notna(), 0.0)

    asset_ret = _to_returns(px)
    gross = position.shift(1).fillna(0.0) * asset_ret  # penundaan satu hari
    net = gross - _apply_costs(position, cfg.cost_bps)

    return pd.DataFrame({"sinyal": position, "return": net.fillna(0.0), "zscore": z})


# =============================================================================
#  AGEN 3: MARKET-NEUTRAL (PAIR TRADING)
# =============================================================================

def pair_trading_agent(
    prices: pd.DataFrame, cfg: AgentConfig, a: str = "BTC", b: str = "ETH"
) -> pd.DataFrame:
    """
    KEGUNAAN  Agen yang memperdagangkan SELISIH antara dua aset, bukan arah
              pasar. Selalu punya satu kaki beli dan satu kaki jual sekaligus,
              sehingga eksposur terhadap arah pasar mendekati nol.

    MASUK     DataFrame harga (T, 2), objek AgentConfig
    KELUAR    DataFrame (T, 5) kolom: sinyal, return, zscore, beta, spread

    TIGA LANGKAH
        1. Hitung beta  = berapa unit aset B untuk menetralkan 1 unit aset A
        2. Hitung spread = log(A) - beta * log(B)
        3. Kalau spread menyimpang jauh, ambil posisi menunggu ia kembali
    """
    # Penjagaan awal. f di depan teks menandakan f-string, yaitu teks yang
    # menyisipkan nilai variabel di dalam kurung kurawal.
    if b not in prices.columns:
        raise ValueError(f"Aset '{b}' tidak tersedia; pair trading butuh dua aset.")

    # Logaritma harga dipakai supaya selisih menjadi RASIO. Dengan begitu
    # harga BTC yang puluhan ribu dan ETH yang ribuan bisa dibandingkan adil.
    log_a = np.log(prices[a])
    log_b = np.log(prices[b])
    ret_a = _to_returns(prices[a])
    ret_b = _to_returns(prices[b])

    bw = cfg.pair_beta_window

    # ===== KEPUTUSAN TEKNIS PENTING =====
    # Beta diestimasi dari RETURN, bukan dari HARGA.
    #
    # Rumusnya regresi linear sederhana: kovarians dibagi varians.
    #
    # Versi pertama program ini memakai harga (log_a.rolling.cov(log_b)), dan
    # hasilnya berayun antara 0.75 sampai 1.87 padahal nilai sebenarnya 1.15.
    # Penyebabnya, harga mengandung tren sehingga non-stasioner, dan regresi
    # atasnya jadi kacau. Return bersifat stasioner sehingga estimasinya stabil.
    #
    # clip(-5, 5) memotong nilai ekstrem: di bawah -5 jadi -5, di atas 5 jadi 5.
    # Ini pengaman agar beta tidak meledak saat varians penyebut mendekati nol.
    beta = (ret_a.rolling(bw).cov(ret_b) / ret_b.rolling(bw).var(ddof=1)).clip(-5, 5)

    win = cfg.pair_window

    # Spread = selisih log-harga setelah dikoreksi hedge ratio.
    # Kalau kedua aset bergerak seiring, spread berayun di sekitar nilai tetap.
    spread = log_a - beta * log_b

    # Jendela beta (250) sengaja DIPISAH dari jendela sinyal (30).
    # Hedge ratio perlu bergerak lambat agar stabil; sinyal perlu responsif.
    ma = spread.rolling(win).mean()
    sd = spread.rolling(win).std(ddof=1)
    z = (spread - ma) / sd.replace(0.0, np.nan)

    position = pd.Series(0.0, index=prices.index)
    position[z < -cfg.pair_entry] = 1.0   # spread terlalu rendah -> long spread
    position[z > cfg.pair_entry] = -1.0   # spread terlalu tinggi -> short spread
    position = position.where(z.notna(), 0.0)

    # DUA Series digeser di sini, bukan satu:
    #   posisi -> alasannya sama seperti agen lain
    #   beta   -> hedge ratio yang dipakai hari ini harus yang diketahui kemarin
    lag_pos = position.shift(1).fillna(0.0)
    lag_beta = beta.shift(1).fillna(0.0)

    # denom menormalkan supaya total eksposur kotor tetap SATU satuan modal.
    # Tanpa ini, agen 3 memakai modal lebih besar daripada dua agen lain,
    # sehingga perbandingan antar agen tidak adil.
    #   lag_beta = 1.2  ->  denom = 1 + 1.2 = 2.2
    denom = (1.0 + lag_beta.abs()).replace(0.0, np.nan)

    # (ret_a - lag_beta * ret_b) menghitung untung rugi DUA kaki sekaligus:
    #   kaki panjang pada aset A
    #   kaki pendek sebesar beta unit pada aset B
    gross = lag_pos * (ret_a - lag_beta * ret_b) / denom

    net = gross - _apply_costs(position, cfg.cost_bps)

    return pd.DataFrame(
        {
            "sinyal": position,
            "return": net.fillna(0.0),
            # Tiga kolom berikut hanya untuk grafik dan penelusuran:
            "zscore": z,
            "beta": beta,
            "spread": spread,
        }
    )


# =============================================================================
#  MENJALANKAN KETIGA AGEN
# =============================================================================

def run_agents(prices: pd.DataFrame, cfg: AgentConfig | None = None) -> dict:
    """
    KEGUNAAN  Menjalankan ketiga agen dan menggabungkan hasilnya.
              INI PINTU MASUK berkas ini; app.py memanggil fungsi ini.

    MASUK     DataFrame harga (T, 2), objek AgentConfig (boleh kosong)
    KELUAR    dict berisi:
                "returns" -> DataFrame (T-warmup, 3) return harian tiap agen
                "signals" -> DataFrame (T-warmup, 3) posisi tiap agen
                "details" -> dict berisi DataFrame lengkap tiap agen, untuk grafik
                "warmup"  -> jumlah hari awal yang dibuang
                "config"  -> konfigurasi yang dipakai
    """
    # Bentuk "a or b" berarti: pakai a kalau a bernilai benar, selain itu b.
    # Jadi kalau pemanggil tidak memberi konfigurasi, dipakai nilai bawaan.
    cfg = cfg or AgentConfig()

    a1 = trend_following_agent(prices, cfg)
    a2 = mean_reversion_agent(prices, cfg)
    # Agen 3 hanya dijalankan kalau kolom ETH ada, supaya program tetap jalan
    # meski CSV hanya berisi satu aset.
    a3 = pair_trading_agent(prices, cfg) if "ETH" in prices.columns else None

    # [a1, a2] + [a3] menyambung dua list jadi satu list berisi tiga DataFrame
    frames = [a1, a2] + ([a3] if a3 is not None else [])
    labels = AGENT_LABELS[: len(frames)]  # ambil sebanyak jumlah agen yang ada

    # [f["return"] for f in frames] disebut LIST COMPREHENSION: ambil kolom
    # "return" dari tiap DataFrame, hasilnya list berisi tiga Series.
    #
    # pd.concat(..., axis=1) menempelkan Series MENYAMPING jadi satu DataFrame.
    # Kalau axis diubah jadi 0, ketiganya ditumpuk KE BAWAH jadi satu kolom
    # panjang, dan itu salah total.
    returns = pd.concat([f["return"] for f in frames], axis=1)
    returns.columns = labels  # ganti nama ketiga kolom sekaligus

    signals = pd.concat([f["sinyal"] for f in frames], axis=1)
    signals.columns = labels

    # ===== MEMBUANG PERIODE PEMANASAN =====
    # Ambil jendela TERPANJANG di antara ketiga agen. Sebelum itu indikator
    # masih NaN dan semua agen berposisi nol, sehingga baris-baris itu akan
    # mencemari estimasi statistik.
    #
    # Pada nilai bawaan: max(200, 20, 250+30) = 280
    warmup = max(
        cfg.sma_slow,
        cfg.mr_window,
        cfg.pair_beta_window + cfg.pair_window,
    )

    # Pengaman: warmup tidak boleh melebihi panjang data dikurangi 30, supaya
    # selalu tersisa minimal 30 hari untuk menghitung mu, sigma, dan rho.
    warmup = min(warmup, len(returns) - 30) if len(returns) > 60 else 0

    # iloc[warmup:] mengambil baris mulai posisi warmup sampai akhir.
    # Inilah sebabnya data 730 hari menyisakan 450 hari efektif.
    returns = returns.iloc[warmup:]
    signals = signals.iloc[warmup:]

    return {
        "returns": returns,
        "signals": signals,
        # dict(zip(a, b)) memasangkan dua list jadi dict:
        #   zip(["x","y"], [1,2]) -> dict {"x": 1, "y": 2}
        "details": dict(zip(labels, frames)),
        "warmup": warmup,
        "config": cfg,
    }


def agent_performance(result: dict) -> pd.DataFrame:
    """
    KEGUNAAN  Meringkas kinerja tiap agen jadi tabel untuk Tab 1.
              MURNI TAMPILAN, tidak dipakai perhitungan lain.

    MASUK     dict hasil run_agents
    KELUAR    DataFrame (3, 6) - satu baris per agen
    """
    rets = result["returns"]
    sig = result["signals"]

    # mean() dan std() pada DataFrame bekerja PER KOLOM, hasilnya Series
    # panjang 3 (satu nilai per agen).
    mu = rets.mean()
    sd = rets.std(ddof=1)

    # ===== MENGHITUNG KURVA EKUITAS DAN DRAWDOWN =====
    # cumprod mengalikan secara kumulatif: hasil[i] = hasil[i-1] * x[i]
    #   rets     : [10, -20, 5]           <- persen
    #   1+r/100  : [1.10, 0.80, 1.05]     <- faktor pertumbuhan harian
    #   cumprod  : [1.10, 0.88, 0.924]    <- nilai modal, awalnya 1.0
    equity = (1.0 + rets / 100.0).cumprod()

    # cummax mencatat nilai TERTINGGI sampai baris itu:
    #   equity : [1.10, 0.88, 0.924]
    #   cummax : [1.10, 1.10, 1.10]
    peak = equity.cummax()

    # Seberapa dalam turun dari puncaknya, diambil yang terdalam
    #   (peak-eq)/peak : [0.0, 0.20, 0.16]  ->  max = 0.20  ->  20%
    mdd = ((peak - equity) / peak).max() * 100.0

    return pd.DataFrame(
        {
            "Return harapan (%/hari)": mu.round(4),
            "Volatilitas (%/hari)": sd.round(4),
            "Sharpe (mu/sigma)": (mu / sd).round(2),
            # (sig != 0) menghasilkan True/False. Dalam numpy, True = 1 dan
            # False = 0, sehingga rata-ratanya = proporsi hari agen berposisi.
            #   sig      : [0, 1, -1, 0, 1]
            #   sig != 0 : [F, T,  T, F, T]
            #   mean()   : 3/5 = 0.6  ->  60%
            "Hari aktif (%)": (sig != 0).mean().mul(100).round(1),
            "Max drawdown (%)": mdd.round(2),
            # Nilai ekuitas terakhir dikurangi 1, dikali 100
            "Total return (%)": ((equity.iloc[-1] - 1) * 100).round(2),
        }
    )
