"""
Unduh harga harian BTC dan ETH dari Binance, simpan ke folder data/.

    python fetch_data.py                        # 730 hari
    python fetch_data.py --days 90              # 90 hari
    python fetch_data.py --days 1000 --out data/prices_1000.csv
"""

import argparse
import sys

from core.agents import AgentConfig, agent_performance, run_agents
from core.binance_api import fetch_and_save


def main() -> int:
    parser = argparse.ArgumentParser(description="Unduh harga BTC/ETH dari Binance.")
    parser.add_argument("--days", type=int, default=730,
                        help="Jumlah hari (maksimum 1000 per permintaan).")
    parser.add_argument("--out", default=None, help="Lokasi CSV keluaran.")
    args = parser.parse_args()

    keluaran = args.out or f"data/prices_{args.days}.csv"
    print(f"Mengunduh {args.days} hari harga BTC dan ETH dari Binance ...\n")

    try:
        harga, path = fetch_and_save(days=args.days, path=keluaran)
    except ConnectionError as exc:
        print("GAGAL\n" + str(exc), file=sys.stderr)
        return 1

    print(f"Tersimpan di : {path.resolve()}")
    print(f"Periode      : {harga.index[0].date()} sampai {harga.index[-1].date()}")
    print(f"Jumlah baris : {len(harga)}\n")

    bawaan = AgentConfig()
    if len(harga) < bawaan.requires_days() + 180:
        cfg = AgentConfig.for_history(len(harga))
        print(
            f"Data lebih pendek dari {bawaan.requires_days()} hari pemanasan, "
            "jendela indikator diperkecil otomatis:\n"
            f"  SMA {cfg.sma_fast}/{cfg.sma_slow}, z-score {cfg.mr_window}, "
            f"beta {cfg.pair_beta_window} + sinyal {cfg.pair_window}"
        )
    else:
        cfg = bawaan
        print("Panjang data mencukupi untuk parameter penuh.")

    hasil = run_agents(harga, cfg)
    print(f"\nPeriode efektif: {len(hasil['returns'])} hari")
    print(agent_performance(hasil).to_string())

    print("\nJalankan `streamlit run app.py` untuk membuka aplikasinya.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
