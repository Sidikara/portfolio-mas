"""
===============================================================================
 binance_api.py  -  MENGUNDUH DATA HARGA
===============================================================================

APA ISI BERKAS INI
    Mengunduh harga historis dari Binance dan menyimpannya ke CSV.
    Berkas ini TIDAK IKUT DALAM PERHITUNGAN APA PUN.

DIPANGGIL OLEH  fetch_data.py, bukan oleh app.py

Pengunduh harga historis dari Binance Public API.

Memakai endpoint /api/v3/klines yang bersifat publik: tanpa API key, tanpa
autentikasi, hanya membaca data pasar.

Hasil unduhan selalu disimpan ke CSV. Saat presentasi, aplikasi membaca CSV
tersebut sehingga tidak bergantung pada koneksi internet.

Binance diblokir pada tingkat DNS di sebagian jaringan Indonesia. Bila terjadi,
ganti DNS ke 1.1.1.1 atau unduh CSV secara manual.
"""

from __future__ import annotations

import json
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

BASE_URLS = [
    "https://data-api.binance.vision",
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
]

SYMBOLS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}
MAX_LIMIT = 1000
UA = "Mozilla/5.0 portfolio-mas/1.0 (akademik)"


def _get_json(url: str, timeout: int = 25):
    """
    KEGUNAAN  Mengirim permintaan HTTP dan membaca balasannya sebagai JSON.

    MASUK     alamat URL
    KELUAR    struktur data Python (list atau dict)

    urllib adalah pustaka BAWAAN Python, jadi tidak perlu memasang paket
    tambahan seperti requests.
    """
    # Konteks SSL untuk koneksi https
    ctx = ssl.create_default_context()
    # headers User-Agent memberi tahu server jenis program yang meminta.
    # Sebagian server menolak permintaan tanpa identitas ini.
    req = urllib.request.Request(url, headers={"User-Agent": UA})

    # Kata kunci "with" memastikan koneksi DITUTUP otomatis setelah selesai,
    # bahkan kalau terjadi kesalahan di tengah jalan.
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        # decode  -> ubah byte mentah jadi teks
        # json.loads -> ubah teks jadi struktur data Python
        return json.loads(resp.read().decode("utf-8"))


def fetch_symbol(symbol: str, interval: str = "1d", limit: int = 1000) -> pd.Series:
    """
    Unduh candle satu simbol, kembalikan deret harga penutupan.

    Struktur balasan klines berupa daftar baris; kolom 0 adalah waktu buka
    dalam milidetik, kolom 4 adalah harga penutupan.
    """
    # Batasi antara 1 dan 1000, karena 1000 adalah batas maksimum satu
    # permintaan pada endpoint klines
    limit = max(1, min(int(limit), MAX_LIMIT))

    # Kumpulkan pesan kesalahan dari tiap host yang gagal, untuk ditampilkan
    # sekaligus kalau semuanya gagal
    kesalahan = []

    for base in BASE_URLS:
        url = f"{base}/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
        try:
            data = _get_json(url)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
            # try/except menangkap kesalahan jaringan. Perintah "continue"
            # melompat ke host berikutnya, TIDAK menghentikan program.
            # Inilah yang membuat program tetap jalan ketika satu alamat
            # diblokir.
            kesalahan.append(f"{base}: {exc}")
            continue

        if not isinstance(data, list) or not data:
            kesalahan.append(f"{base}: balasan kosong")
            continue

        # Balasan klines berupa daftar baris, tiap baris berisi 12 nilai:
        #   kolom 0  = waktu buka (milidetik)
        #   kolom 1-3 = harga buka, tertinggi, terendah
        #   kolom 4  = HARGA PENUTUPAN  <- yang kita ambil
        #   kolom 5+ = volume dan lainnya
        #
        # [b[0] for b in data] disebut LIST COMPREHENSION: cara ringkas
        # menulis perulangan yang membangun daftar.
        # unit="ms" memberi tahu pandas angkanya dalam milidetik, bukan detik.
        waktu = pd.to_datetime([b[0] for b in data], unit="ms")
        tutup = [float(b[4]) for b in data]
        seri = pd.Series(tutup, index=waktu, name=symbol)
        seri.index = seri.index.normalize()
        seri.index.name = "Tanggal"
        return seri

    raise ConnectionError(
        "Gagal mengunduh dari seluruh host Binance.\n  "
        + "\n  ".join(kesalahan)
        + "\n\nKemungkinan penyebab: Binance diblokir DNS pada jaringan ini. "
        "Ganti DNS ke 1.1.1.1, atau unduh CSV secara manual dengan kolom "
        "Tanggal, BTC, ETH lalu simpan ke folder data/."
    )


def fetch_prices(days: int = 730, interval: str = "1d") -> pd.DataFrame:
    """Unduh BTC dan ETH sekaligus, gabungkan menjadi satu tabel harga."""
    kolom = []
    # SYMBOLS.items() menelusuri pasangan kunci dan nilai pada dict:
    #   "BTC" dengan "BTCUSDT", lalu "ETH" dengan "ETHUSDT"
    for nama_pendek, simbol in SYMBOLS.items():
        seri = fetch_symbol(simbol, interval=interval, limit=days)
        seri.name = nama_pendek
        kolom.append(seri.to_frame())
        # Tunggu 0.3 detik antar permintaan supaya tidak kena pembatasan
        # laju dari server
        time.sleep(0.3)

    harga = pd.concat(kolom, axis=1).dropna()
    harga.index.name = "Tanggal"
    return harga


def fetch_and_save(
    days: int = 730, path: str | Path = "data/prices_730.csv", interval: str = "1d"
) -> tuple[pd.DataFrame, Path]:
    """Unduh lalu simpan ke CSV."""
    path = Path(path)
    # mkdir membuat folder data/ kalau belum ada.
    #   parents=True   -> buat folder induknya sekaligus
    #   exist_ok=True  -> jangan error kalau foldernya sudah ada
    path.parent.mkdir(parents=True, exist_ok=True)

    harga = fetch_prices(days=days, interval=interval)
    harga.to_csv(path)
    return harga, path
