"""
===============================================================================
 app.py  -  ANTARMUKA STREAMLIT
===============================================================================

APA ISI BERKAS INI
    TIDAK ADA LOGIKA PERHITUNGAN DI SINI. Seluruh isinya memanggil fungsi
    dari folder core/, lalu menampilkan hasilnya sebagai tabel dan grafik.

    Kalau ingin mengubah PERHITUNGAN, jangan sentuh berkas ini.
    Kalau ingin mengubah TAMPILAN, di sinilah tempatnya.

BAGIAN YANG BENAR-BENAR MENGHITUNG hanya 8 baris, ada di bagian
"PIPELINE INTI" di bawah. Sisanya (sekitar 400 baris) hanya menampilkan.

SUSUNAN BERKAS
    1. Sidebar          pilih data dan atur parameter
    2. Pipeline inti    8 baris yang menjalankan seluruh perhitungan
    3. Tab 1            Tier 1: harga, kinerja agen, sinyal
    4. Tab 2            Tier 2: statistik, tabel skenario, CVaR
    5. Tab 3            efficient frontier
    6. Tab 4            Tier 3: HHI, jaringan agen, risiko ekor
    7. Tab 5            sensitivitas lambda
===============================================================================

Portfolio Optimization Dalam Sistem Perdagangan Multi-Agen

Arsitektur tiga tier:

    TIER 1  Agen Strategi     tiga agen menghasilkan sinyal dari harga
    TIER 2  Koordinator       mengalokasikan modal antar agen
    TIER 3  Pemantau Risiko   mengukur konsentrasi dan keterkaitan

Sumber data tunggal: berkas CSV harga historis Binance di folder data/.

Jalankan:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.agents import AgentConfig, agent_performance, run_agents
from core.datasets import (
    daftar_dataset,
    from_agent_returns,
    load_prices,
    price_summary,
    summary_table,
)
from core.optimizer import calibrate_lambda, cvar_optimize, max_sharpe, mean_variance, min_variance, random_portfolios
from core.risk import historical_var_cvar, risk_report
from core.scenarios import (
    build_scenarios,
    coordination_benefit,
    risk_benefit,
    scenarios_dataframe,
)
from core.statistics import condition_number, covariance_matrix, shrink_covariance

st.set_page_config(page_title="Portfolio Optimization Multi-Agen", layout="wide")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("Panel Kontrol")

    st.subheader("Data Harga")
    dataset = daftar_dataset()
    if not dataset:
        st.error(
            "Folder data/ kosong. Jalankan `python fetch_data.py` terlebih dahulu."
        )
        st.stop()

    nama_berkas = st.selectbox("Berkas CSV", list(dataset.keys()))
    prices = load_prices(dataset[nama_berkas])
    st.caption(
        f"{len(prices)} hari, {prices.index[0].date()} s/d {prices.index[-1].date()}"
    )

    st.divider()
    st.subheader("Parameter Agen (Tier 1)")

    bawaan = AgentConfig()
    pendek = len(prices) < bawaan.requires_days() + 180
    saran = AgentConfig.for_history(len(prices)) if pendek else bawaan

    if pendek:
        st.warning(
            f"Data {len(prices)} hari, sedangkan parameter penuh menuntut "
            f"{bawaan.requires_days()} hari pemanasan. Jendela indikator "
            "diperkecil otomatis."
        )

    c1, c2 = st.columns(2)
    sma_fast = c1.number_input("SMA cepat", 3, 120, saran.sma_fast)
    sma_slow = c2.number_input("SMA lambat", 10, 400, saran.sma_slow, step=5)
    mr_window = c1.number_input("Jendela z-score", 5, 120, saran.mr_window)
    mr_entry = c2.number_input("Ambang z", 0.5, 3.0, 1.0, step=0.25)
    pair_window = c1.number_input("Jendela pair", 5, 180, saran.pair_window, step=5)
    pair_beta = c2.number_input("Jendela beta", 20, 500, saran.pair_beta_window, step=10)
    cost_bps = st.number_input("Biaya transaksi (bps)", 0.0, 50.0, 5.0, step=1.0)

    st.divider()
    st.subheader("Parameter Koordinator (Tier 2)")

    w_max = st.slider("Batas posisi maksimum", 0.34, 1.0, 0.50, step=0.01)
    shrink = st.slider(
        "Shrinkage kovarians", 0.0, 1.0, 0.0, step=0.05,
        help="Menstabilkan matriks kovarians saat jumlah observasi sedikit.",
    )
    alpha = st.slider("Tingkat kepercayaan CVaR", 0.80, 0.99, 0.95, step=0.01)


# =============================================================================
#  PIPELINE INTI
#  Delapan baris berikut adalah KESELURUHAN perhitungan program.
#  Sisanya hanya menampilkan hasilnya.
#
#      harga  ->  agen  ->  statistik  ->  kovarians  ->  bobot
# =============================================================================
cfg = AgentConfig(
    sma_fast=int(sma_fast),
    sma_slow=int(sma_slow),
    mr_window=int(mr_window),
    mr_entry=float(mr_entry),
    pair_window=int(pair_window),
    pair_beta_window=int(pair_beta),
    cost_bps=float(cost_bps),
)

# TIER 1: jalankan ketiga agen pada deret harga -> return harian tiap agen
hasil_agen = run_agents(prices, cfg)

# Hitung mu, sigma, dan rho dari return tersebut
data = from_agent_returns(hasil_agen["returns"], source=nama_berkas)

# Bangun matriks kovarians, lalu stabilkan bila shrinkage diaktifkan
cov = shrink_covariance(covariance_matrix(data.sigma, data.rho), shrink)

SHORT_NAMES = [f"Agen {i + 1}" for i in range(data.n_agents)]
# Periksa apakah ADA SETIDAKNYA SATU agen dengan return harapan positif.
# Kalau tidak ada, skenario S3 Max-Sharpe tidak terdefinisi dan dihilangkan.
ada_alpha = bool(np.any(data.mu > 0))

# lam_star dipakai tab Sensitivitas sebagai titik tengah sapuan lambda
lam_star = None
if ada_alpha:
    kandidat = calibrate_lambda(data.mu, cov, w_max=w_max)
    # lambda* = S / (2 sigma_p). Bila Sharpe portofolio terbaik masih nol atau
    # negatif, lambda* tidak positif dan sapuan logaritmik tidak terdefinisi.
    lam_star = kandidat if kandidat > 1e-9 else None

# Benchmark buy-and-hold sebagai pembanding jujur.
# reindex menyelaraskan tanggal benchmark dengan tanggal return agen, supaya
# perbandingannya mencakup periode yang SAMA PERSIS. Tanpa ini, benchmark
# akan mencakup periode pemanasan yang tidak dialami agen.
benchmark_ret = (prices["BTC"].pct_change() * 100.0).reindex(data.returns.index)

# TIER 2: bangun keempat skenario alokasi beserta metriknya
rows = build_scenarios(
    data.mu, cov, w_max=w_max, agent_names=data.names,
    returns=data.returns, benchmark=benchmark_ret,
)

st.title("Portfolio Optimization Dalam Sistem Perdagangan Multi-Agen")
st.caption(
    f"Data: {nama_berkas} | {len(prices)} hari harga | "
    f"{len(data.returns)} hari efektif setelah pemanasan indikator"
)

if not ada_alpha:
    st.warning(
        "Tidak ada agen dengan return harapan positif pada periode ini. "
        "Portofolio Sharpe-maksimum (S3) tidak terdefinisi dan dihilangkan. "
        "Analisis memakai S4 Min-Variance, yang hanya bersandar pada matriks "
        "kovarians sehingga tetap bermakna."
    )

if condition_number(cov) > 1e4:
    st.warning(
        f"Condition number kovarians {condition_number(cov):,.0f}. "
        "Matriks hampir singular. Naikkan shrinkage atau perpanjang periode."
    )

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["1. Agen (Tier 1)", "2. Alokasi (Tier 2)", "3. Efficient Frontier",
     "4. Risiko (Tier 3)", "5. Sensitivitas"]
)


# =============================================================================
#  TAB 1  -  TIER 1: AGEN STRATEGI
#  Menampilkan deret harga, kinerja ketiga agen, dan visualisasi sinyalnya.
#  Ini bukti bahwa agen benar-benar bekerja, bukan angka asumsi.
# =============================================================================
with tab1:
    st.subheader("Deret Harga")
    st.dataframe(price_summary(prices), use_container_width=True)

    norm = prices / prices.iloc[0] * 100.0
    fig = px.line(norm, labels={"value": "Indeks (hari pertama = 100)"},
                  color_discrete_sequence=["#F58518", "#4C78A8"])
    fig.update_layout(height=320, legend_title_text="", margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Kinerja Tiga Agen")
    st.caption(
        f"Pemanasan indikator {hasil_agen['warmup']} hari. "
        "Posisi hari t ditentukan dari informasi sampai t, return "
        "direalisasikan t+1 untuk mencegah look-ahead bias."
    )
    st.dataframe(agent_performance(hasil_agen), use_container_width=True)

    eq = (1.0 + data.returns / 100.0).cumprod()
    fig = px.line(eq, labels={"value": "Ekuitas (x modal awal)"})
    fig.update_layout(height=340, legend_title_text="", margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Sinyal Agen")

    pilihan = st.selectbox("Agen", list(hasil_agen["details"].keys()))
    detail = hasil_agen["details"][pilihan].loc[data.returns.index]

    fig = go.Figure()
    if "sma_cepat" in detail:
        fig.add_trace(go.Scatter(x=prices.index, y=prices["BTC"], name="BTC",
                                 line=dict(color="#9AA0A6", width=1)))
        fig.add_trace(go.Scatter(x=detail.index, y=detail["sma_cepat"],
                                 name=f"SMA {cfg.sma_fast}", line=dict(color="#4C78A8")))
        fig.add_trace(go.Scatter(x=detail.index, y=detail["sma_lambat"],
                                 name=f"SMA {cfg.sma_slow}", line=dict(color="#E45756")))
        fig.update_layout(yaxis_title="Harga BTC")
    else:
        ambang = cfg.pair_entry if "beta" in detail else cfg.mr_entry
        fig.add_trace(go.Scatter(x=detail.index, y=detail["zscore"], name="z-score",
                                 line=dict(color="#4C78A8")))
        fig.add_hline(y=ambang, line_dash="dash", line_color="#E45756")
        fig.add_hline(y=-ambang, line_dash="dash", line_color="#54A24B")
        fig.add_hline(y=0, line_color="#9AA0A6", line_width=1)
        fig.update_layout(yaxis_title="z-score")

    fig.update_layout(height=320, legend_title_text="", margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)

    fig = px.area(hasil_agen["signals"][pilihan], labels={"value": "Posisi"})
    fig.update_layout(height=180, showlegend=False, margin=dict(t=10),
                      yaxis=dict(range=[-1.2, 1.2]))
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
#  TAB 2  -  TIER 2: ALOKASI MODAL
#  Tab paling penting. Menampilkan tabel perbandingan keempat skenario dan
#  metrik manfaat koordinasi.
# =============================================================================
with tab2:
    st.subheader("Statistik Masukan Koordinator")

    k1, k2 = st.columns([1, 1])
    with k1:
        st.dataframe(summary_table(data), use_container_width=True)
    with k2:
        fig = px.imshow(np.round(data.rho, 3), x=SHORT_NAMES, y=SHORT_NAMES,
                        text_auto=True, color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1, aspect="auto")
        fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0),
                          title_text="Matriks korelasi")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Perbandingan Skenario Alokasi")
    st.dataframe(scenarios_dataframe(rows), use_container_width=True, hide_index=True)

    manfaat_risiko = risk_benefit(rows, "S4")
    if manfaat_risiko:
        st.markdown(
            f"**Manfaat koordinasi pada dimensi risiko** "
            f"(S4 vs {manfaat_risiko['pembanding']})"
        )
        r1, r2, r3 = st.columns(3)
        r1.metric("Volatilitas S4", f"{manfaat_risiko['vol_portofolio']:.4f}",
                  f"{manfaat_risiko['vol_change_pct']:+.1f}%", delta_color="inverse")
        r2.metric("vs rata-rata agen", f"{manfaat_risiko['vol_rata_agen']:.4f}",
                  f"{manfaat_risiko['vol_vs_rata_pct']:+.1f}%", delta_color="inverse")
        if "dd_portofolio" in manfaat_risiko and not np.isnan(
            manfaat_risiko.get("dd_vs_rata_pct", np.nan)
        ):
            r3.metric("Max drawdown S4", f"{manfaat_risiko['dd_portofolio']:.1f}%",
                      f"{manfaat_risiko['dd_vs_rata_pct']:+.1f}%", delta_color="inverse")
        st.caption(
            "Pembanding dipilih agen tunggal dengan volatilitas TERENDAH, "
            "bukan yang paling mudah dikalahkan."
        )

    manfaat_return = coordination_benefit(rows, data.mu)
    if manfaat_return:
        st.divider()
        st.markdown(
            f"**Manfaat koordinasi pada dimensi return** "
            f"(S3 vs {manfaat_return['benchmark']})"
        )
        m1, m2 = st.columns(2)
        m1.metric("Sharpe S3", f"{manfaat_return['sharpe_optimized']:.3f}",
                  f"{manfaat_return['sharpe_gain_pct']:+.1f}%")
        m2.metric("Volatilitas", f"{manfaat_return['vol_optimized']:.4f}",
                  f"{manfaat_return['vol_change_pct']:+.1f}%", delta_color="inverse")

    st.divider()
    st.subheader("Komposisi Bobot")
    comp = pd.DataFrame(
        {r["label"]: r["weights"] for r in rows if r["kelompok"] == "portofolio"},
        index=SHORT_NAMES,
    ).T
    fig = px.bar(comp, barmode="stack", labels={"value": "Bobot", "index": ""})
    fig.update_layout(height=300, legend_title_text="", margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Optimisasi CVaR")
    cv = cvar_optimize(data.returns.to_numpy(), data.mu, cov, alpha=alpha, w_max=w_max)
    v1, v2, v3 = st.columns(3)
    v1.metric(f"VaR {alpha:.0%}", f"{cv['var']:.4f} %/hari")
    v2.metric(f"CVaR {alpha:.0%}", f"{cv['cvar']:.4f} %/hari")
    v3.metric("Sharpe", f"{cv['sharpe']:.3f}")
    st.write(
        "Bobot CVaR-optimal: "
        + ", ".join(f"{n} = {v:.3f}" for n, v in zip(SHORT_NAMES, cv["weights"]))
    )
    st.caption(
        "CVaR meminimalkan rata-rata kerugian pada ekor distribusi, "
        "berbeda dari mean-variance yang menimbang seluruh sebaran."
    )


# =============================================================================
#  TAB 3  -  EFFICIENT FRONTIER
#  Menggambar 4000 portofolio acak sebagai latar, lalu menyorot posisi
#  skenario S1, S2, S3, dan S4 di antaranya.
# =============================================================================
with tab3:
    st.subheader("Efficient Frontier")
    st.caption(
        "Titik berwarna adalah portofolio acak. Batas kiri atas awan titik "
        "adalah kombinasi paling efisien."
    )

    cloud = random_portfolios(data.mu, cov, n_samples=4000)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cloud["volatility"], y=cloud["return"], mode="markers",
        name="Portofolio acak",
        marker=dict(size=4, color=cloud["sharpe"], colorscale="Viridis",
                    opacity=0.45, colorbar=dict(title="Sharpe")),
        hovertemplate="Vol %{x:.4f}<br>Return %{y:.4f}<extra></extra>",
    ))

    sorotan = [("S4 Min-Variance", min_variance(cov, data.mu, w_max=w_max),
                "#4C78A8", "diamond")]
    if ada_alpha:
        sorotan.insert(0, ("S3 Max-Sharpe", max_sharpe(data.mu, cov, w_max=w_max),
                           "#E45756", "star"))
    for r in rows:
        if r["label"].startswith(("S1", "S2")):
            sorotan.append((r["label"], r, "#F58518", "circle"))

    for label, m, warna, bentuk in sorotan:
        fig.add_trace(go.Scatter(
            x=[m["volatility"]], y=[m["return"]], mode="markers+text",
            name=label, text=[label], textposition="top center",
            marker=dict(size=15, color=warna, symbol=bentuk,
                        line=dict(width=1, color="white")),
        ))

    fig.update_layout(xaxis_title="Volatilitas portofolio (%/hari)",
                      yaxis_title="Return harapan (%/hari)",
                      height=540, margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
#  TAB 4  -  TIER 3: RISIKO KONSENTRASI
#  Menampilkan HHI, effective N, grafik jaringan antar agen, dan risiko ekor.
# =============================================================================
with tab4:
    st.subheader("Konsentrasi dan Keterkaitan")

    terpilih = next(
        (r for r in rows if r["label"].startswith("S3")),
        next(r for r in rows if r["label"].startswith("S4")),
    )
    st.caption(f"Portofolio yang dinilai: {terpilih['label']}")

    lap = risk_report(terpilih["weights"], data.sigma, data.rho, cov)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("HHI", f"{lap['HHI']:.3f}", help="1/N bila merata, 1 bila terpusat")
    c2.metric("Effective N", f"{lap['Effective N']:.2f}", help="1/HHI")
    c3.metric("Korelasi tertimbang", f"{lap['Weighted correlation']:.3f}")
    c4.metric("Diversification ratio", f"{lap['Diversification ratio']:.3f}")

    st.divider()
    hhi_df = pd.DataFrame({
        "Skenario": [r["label"] for r in rows if not np.isnan(r.get("hhi", np.nan))],
        "HHI": [r["hhi"] for r in rows if not np.isnan(r.get("hhi", np.nan))],
    })
    fig = px.bar(hhi_df, x="Skenario", y="HHI")
    fig.add_hline(y=1.0 / data.n_agents, line_dash="dash",
                  annotation_text="Batas bawah (alokasi merata)")
    fig.update_layout(height=340, xaxis_title="", margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("**Jaringan Keterkaitan Agen**")
    st.caption("Ukuran simpul mencerminkan bobot modal, tebal sisi mencerminkan korelasi.")

    n = data.n_agents
    sudut = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / 2
    xs, ys = np.cos(sudut), np.sin(sudut)
    w = terpilih["weights"]

    net = go.Figure()
    for i in range(n):
        for j in range(i + 1, n):
            net.add_trace(go.Scatter(
                x=[xs[i], xs[j]], y=[ys[i], ys[j]], mode="lines",
                line=dict(width=1 + 8 * abs(data.rho[i, j]),
                          color="rgba(120,120,120,0.5)"),
                hoverinfo="text", text=f"rho = {data.rho[i, j]:.2f}", showlegend=False,
            ))
    net.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text",
        text=[f"{s}<br>w={v:.2f}" for s, v in zip(SHORT_NAMES, w)],
        textposition="bottom center",
        marker=dict(size=30 + 110 * w, color="#4C78A8", opacity=0.85),
        showlegend=False,
    ))
    net.update_layout(height=400, xaxis=dict(visible=False, range=[-1.6, 1.6]),
                      yaxis=dict(visible=False, range=[-1.6, 1.6], scaleanchor="x"),
                      margin=dict(t=10, b=10))
    st.plotly_chart(net, use_container_width=True)

    st.divider()
    st.markdown("**Risiko Ekor Empiris**")
    var_e, cvar_e = historical_var_cvar(data.returns.to_numpy(), w, alpha=alpha)
    d1, d2, d3 = st.columns(3)
    d1.metric(f"VaR {alpha:.0%}", f"{var_e:.4f} %")
    d2.metric(f"CVaR {alpha:.0%}", f"{cvar_e:.4f} %")
    d3.metric("Max drawdown", f"{terpilih.get('drawdown', float('nan')):.2f} %")


# =============================================================================
#  TAB 5  -  SENSITIVITAS LAMBDA
#  Menyapu nilai lambda dan menggambar pengaruhnya pada Sharpe, HHI, dan
#  komposisi bobot. Inilah yang menunjukkan masalah "solusi sudut".
# =============================================================================
with tab5:
    st.subheader("Sensitivitas terhadap Risk Aversion (lambda)")

    if lam_star is None:
        st.info(
            "Kalibrasi lambda mengikuti lambda* = S / (2 sigma_p), sehingga "
            "memerlukan Sharpe ratio portofolio yang positif. Pada periode ini "
            "tidak ada portofolio dengan Sharpe positif, jadi sapuan lambda "
            "tidak terdefinisi. Gunakan tab Alokasi untuk menilai S4 "
            "Min-Variance, yang tidak memiliki parameter lambda."
        )
        st.stop()

    st.caption(
        f"lambda terkalibrasi: {lam_star:,.2f}, mengikuti lambda* = S / (2 sigma_p). "
        "Nilai lambda yang terlalu kecil menghasilkan solusi sudut."
    )

    grid = np.logspace(np.log10(max(lam_star * 0.01, 1e-3)),
                       np.log10(lam_star * 30), 40)
    catatan = []
    for lam in grid:
        try:
            r = mean_variance(data.mu, cov, lam=float(lam), w_max=w_max)
        except (RuntimeError, ValueError):
            continue
        baris = {"lambda": lam, "Sharpe": r["sharpe"],
                 "HHI": float(np.sum(r["weights"] ** 2))}
        for k, nama in enumerate(SHORT_NAMES):
            baris[nama] = r["weights"][k]
        catatan.append(baris)

    sens = pd.DataFrame(catatan)

    s1, s2 = st.columns(2)
    with s1:
        fig = px.line(sens, x="lambda", y=["Sharpe"], log_x=True)
        fig.add_vline(x=lam_star, line_dash="dash", annotation_text="lambda*")
        fig.update_layout(height=310, yaxis_title="Sharpe", legend_title_text="",
                          margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)
    with s2:
        fig = px.line(sens, x="lambda", y=["HHI"], log_x=True)
        fig.add_hline(y=1.0 / data.n_agents, line_dash="dot",
                      annotation_text="alokasi merata")
        fig.add_vline(x=lam_star, line_dash="dash", annotation_text="lambda*")
        fig.update_layout(height=310, yaxis_title="HHI", legend_title_text="",
                          margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Pergeseran Bobot terhadap lambda**")
    fig = px.area(sens, x="lambda", y=SHORT_NAMES, log_x=True)
    fig.add_vline(x=lam_star, line_dash="dash", annotation_text="lambda*")
    fig.update_layout(height=360, yaxis_title="Bobot", legend_title_text="",
                      margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)
