"""Developer view: render() is called from app.py."""
import json
import subprocess
import sys
from datetime import datetime

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from common import ROOT, artifacts_ready, prepared
from irrigation_ml.config import ARTIFACTS, PROCESSED


def render():
    st.title("Smart Irrigation: developer console")
    if not artifacts_ready():
        st.error("No artifacts. Run `PYTHONPATH=src python -m irrigation_ml.pipeline`.")
        st.stop()

    cal = json.loads((ARTIFACTS / "calibration.json").read_text())
    tabs = st.tabs(["Overview", "Data & trust", "Forecast", "Dry-risk classifier", "Decision backtest", "Explain", "Case inspector", "Pipeline"])


    @st.cache_data
    def read_oof(name):
        return pd.read_parquet(ARTIFACTS / name)


    @st.cache_data
    def hourly():
        return pd.read_parquet(PROCESSED / "hourly_model_table.parquet")


    # ---------------------------------------------------------------- Overview
    with tabs[0]:
        fm = pd.read_csv(ARTIFACTS / "forecast_metrics.csv")
        allm = fm[fm.segment == "all"].set_index("H")
        c = st.columns(3)
        for col, H in zip(c, (24, 48, 72)):
            col.metric(f"Forecast MAE @ {H} h", f"{allm.loc[H, 'MAE_model']:.2f}", f"{-allm.loc[H, 'skill_MAE']:.0%} vs persistence ({allm.loc[H, 'MAE_persist']:.2f})", delta_color="inverse")
        st.markdown("""
    **Reading guide.** Skill = 1 − MAE_model / MAE_persistence on walk-forward folds (2025-04 → 2025-10, train purged by H).
    Model selection used the same folds, so margins are slightly optimistic. Risk labels use a per-field sensor-calibrated trigger,
    not ground-truth irrigation events, and the rain forecast is ERA5 + simulated noise.""")
        th = pd.DataFrame(cal["thresholds"]).T
        st.subheader("Per-field dry trigger (probe scale)")
        st.dataframe(th.round(2), width="stretch")
        st.caption(f"Data ends {cal['data_end']}. Interval offsets (conformal, points of moisture): {cal['interval_offsets']}")

    # ---------------------------------------------------------------- Data & trust
    with tabs[1]:
        h = hourly()
        st.subheader("Coverage and anomaly rates by field")
        g = h.groupby("field_id").agg(rows=("timestamp", "size"), valid_sm=("is_sm_valid", "mean"), iforest=("anom_iforest", "mean"),
                                      rules=("anom_sm_any", "mean"), dead=("sm_dead", "mean"), battery_med=("battery_voltage", "median"))
        st.dataframe(g.style.format({"valid_sm": "{:.1%}", "iforest": "{:.1%}", "rules": "{:.1%}", "dead": "{:.1%}", "battery_med": "{:.2f}"}), width="stretch")
        st.subheader("Battery voltage vs anomaly rate (why battery is a trust indicator)")
        bins = pd.cut(h["battery_voltage"], [0, 3.6, 3.8, 4.0, 4.1, 4.3])
        bt = h.groupby(bins, observed=True).agg(n=("anom_sm_any", "size"), anomaly_rate=("anom_sm_any", "mean"), dead_rate=("sm_dead", "mean")).reset_index()
        bt["battery_voltage"] = bt["battery_voltage"].astype(str)
        st.altair_chart(alt.Chart(bt).mark_bar().encode(x="battery_voltage:N", y=alt.Y("anomaly_rate:Q", axis=alt.Axis(format="%")), tooltip=["n", "anomaly_rate", "dead_rate"]).properties(height=240), width="stretch")
        st.subheader("Soil-moisture series with masked data")
        f = st.selectbox("Field", sorted(h.field_id.unique()), key="dq_field")
        d = h[h.field_id == f].set_index("timestamp").sort_index()
        daily = d[["sm_clean"]].resample("D").mean().reset_index()
        flagged = d["anom_sm_any"].resample("D").mean().reset_index()
        cc = alt.Chart(daily).mark_line().encode(x="timestamp:T", y=alt.Y("sm_clean:Q", scale=alt.Scale(zero=False)))
        st.altair_chart(cc.properties(height=240), width="stretch")
        st.altair_chart(alt.Chart(flagged).mark_area(color="#c0392b", opacity=.5).encode(x="timestamp:T", y=alt.Y("anom_sm_any:Q", title="daily anomaly share")).properties(height=100), width="stretch")
        st.caption("Gaps in the line are real missing/untrusted data: no interpolation is applied to the target.")
        st.subheader("Sensor drift (qa_drift.csv)")
        qd = PROCESSED / "qa_drift.csv"
        if qd.exists():
            st.dataframe(pd.read_csv(qd).style.apply(lambda r: ["background-color: rgba(231, 76, 60, 0.30)" if r.drift_confirmed else "" for _ in r], axis=1), width="stretch")
            st.caption("CONTROL rows compare two reference products: drift there means part of the apparent drift is in the references, not the probe.")

    # ---------------------------------------------------------------- Forecast
    with tabs[2]:
        st.subheader("Walk-forward metrics vs persistence")
        st.dataframe(fm.style.format({c: "{:.3f}" for c in fm.columns if c not in ("H", "segment", "n")}), width="stretch")
        ab = ARTIFACTS / "forecast_ablation.csv"
        if ab.exists():
            st.subheader("Ablation (earlier run: L1 loss, larger trees; use for relative comparison only)")
            a = pd.read_csv(ab); a = a[a.segment == "all"]
            st.dataframe(a.pivot(index="config", columns="H", values="skill_MAE").style.format("{:+.1%}"), width="stretch")
            st.caption("sensor = probe+trust only; weather = +weather/rain history; forecast = +rain forecast; retro = +ERA5/SMAP (not available live). hl = recency half-life (days).")
        oof = read_oof("forecast_oof.parquet")
        st.subheader("Out-of-fold predictions")
        c1, c2, c3 = st.columns(3)
        H = c1.selectbox("Horizon", [24, 48, 72]); fld = c2.selectbox("Field", sorted(oof.field_id.unique()), key="fc_f")
        fold = c3.selectbox("Fold", sorted(oof.fold.unique()))
        s = oof[(oof.H == H) & (oof.field_id == fld) & (oof.fold == fold)].sort_values("timestamp")
        if len(s):
            m = s.melt("timestamp", ["y", "model", "persist"], var_name="series")
            band = alt.Chart(s).mark_area(opacity=.2).encode(x="timestamp:T", y="q10:Q", y2="q90:Q")
            lines = alt.Chart(m).mark_line().encode(x="timestamp:T", y=alt.Y("value:Q", scale=alt.Scale(zero=False)), color="series:N")
            st.altair_chart((band + lines).properties(height=300), width="stretch")
            st.caption(f"n={len(s)}  MAE model {np.abs(s.model - s.y).mean():.2f} vs persistence {np.abs(s.persist - s.y).mean():.2f}; band = raw (uncalibrated) q10–q90.")
        st.subheader("Error vs realised change")
        oof["chg"] = (oof.y - oof.sm0).abs()
        e = oof.assign(bin=pd.cut(oof.chg, [-.01, .5, 1, 3, 10, 100])).groupby(["H", "bin"], observed=True).apply(
            lambda x: pd.Series({"n": len(x), "MAE_model": (x.model - x.y).abs().mean(), "MAE_persist": (x.persist - x.y).abs().mean()})).reset_index()
        e["bin"] = e["bin"].astype(str)
        st.altair_chart(alt.Chart(e.melt(["H", "bin", "n"], ["MAE_model", "MAE_persist"])).mark_bar().encode(
            x="bin:N", y="value:Q", color="variable:N", xOffset="variable:N", column="H:N").properties(height=200, width=180), width="content")
        st.caption("The model gains on small/medium changes; large rain-driven jumps remain hard.")

    # ---------------------------------------------------------------- Risk
    with tabs[3]:
        from sklearn.metrics import precision_recall_curve, average_precision_score
        r = read_oof("risk_oof.parquet")
        c1, c2 = st.columns(2)
        Hr = c1.selectbox("Horizon", [24, 48, 72], key="rk_h")
        only_ant = c2.checkbox("Anticipation only (probe currently above trigger)", True)
        g = r[(r.H == Hr)]
        g = g[g.gap > 0] if only_ant else g
        pr, rc, th_ = precision_recall_curve(g.y, g.p)
        st.metric("PR-AUC", f"{average_precision_score(g.y, g.p):.3f}", f"base rate {g.y.mean():.3f} · n={len(g)} · positives={int(g.y.sum())}", delta_color="off")
        pc = pd.DataFrame({"recall": rc[:-1], "precision": pr[:-1], "threshold": th_})
        st.altair_chart(alt.Chart(pc).mark_line().encode(x="recall:Q", y=alt.Y("precision:Q", scale=alt.Scale(domain=[0, 1])), tooltip=["threshold", "recall", "precision"]).properties(height=280), width="stretch")
        rows = []
        for tau in (0.5, 0.25, 0.17, 0.1, 0.05, 0.02):
            f = g.p >= tau; tp = int((f & (g.y == 1)).sum()); fp = int((f & (g.y == 0)).sum()); fn = int(((~f) & (g.y == 1)).sum())
            rows.append(dict(tau=tau, c_fn_equiv=round(1 / tau - 1, 1), flagged=int(f.sum()), TP=tp, FP=fp, FN=fn, recall=tp / max(tp + fn, 1), precision=tp / max(tp + fp, 1)))
        st.dataframe(pd.DataFrame(rows).style.format({"recall": "{:.1%}", "precision": "{:.1%}"}), width="stretch")
        cb = g.assign(bin=pd.cut(g.p, [0, .02, .05, .1, .25, .5, 1], include_lowest=True)).groupby("bin", observed=True).agg(n=("y", "size"), mean_p=("p", "mean"), observed=("y", "mean")).reset_index()
        cb["bin"] = cb["bin"].astype(str)
        st.subheader("Calibration")
        st.altair_chart(alt.Chart(cb.melt("bin", ["mean_p", "observed"])).mark_bar().encode(x="bin:N", y="value:Q", color="variable:N", xOffset="variable:N").properties(height=240), width="stretch")
        st.caption("Per-hour rows are strongly autocorrelated: independent events are far fewer than rows (31 dry-down events on 11 distinct days in the 06:00 backtest).")

        om = ARTIFACTS / "offline_metrics.csv"
        if om.exists():
            st.subheader("No-probe fallback (used while the probe is down)")
            o = pd.read_csv(om)
            st.dataframe(o.style.format({c: "{:.3f}" for c in o.columns if c not in ("H", "outage_age", "n", "events", "hard_n", "hard_events")}), width="stretch")
            st.caption("Trained on simulated outages (6 h to 25 d). 'hard' = the hidden probe is still above the trigger. AUC_stale_rule is the naive "
                       "'last reading below trigger' rule. Skill fades with outage length: the app offers an estimate only up to 10 days "
                       "(hard-case AUC is ~0.5 beyond that at 24 h). Walk-forward, trigger from pre-test data.")

    # ---------------------------------------------------------------- Backtest
    with tabs[4]:
        st.caption("Replays the policy on out-of-fold forecasts at 00/06/12/18 h and scores it against what the probe actually did (min reading within 72 h < trigger).")
        cfn = st.select_slider("Cost of a missed irrigation (c_fn, c_fp = 1)", [1, 2, 3, 5, 10, 20, 50], value=5)

        @st.cache_data(show_spinner="Running backtest...")
        def bt(c):
            from irrigation_ml.backtest import run
            from irrigation_ml.decision import Policy
            return run(Policy(c_fn=c))[0]
        cs = bt(float(cfn))
        from irrigation_ml.backtest import score
        ant = cs[~cs.naive]
        flag = lambda g: g.action.isin(["irrigate", "irrigate_soon"])
        res = {"naive: sm0 < trigger (anticipation)": score(ant, ant.naive), "ML irrigate (anticipation)": score(ant, ant.action == "irrigate"),
               "ML irrigate or plan (anticipation)": score(ant, flag(ant)), "ML irrigate or plan (all cases)": score(cs, flag(cs))}
        st.dataframe(pd.DataFrame(res).T.style.format({"recall": "{:.1%}", "precision": "{:.1%}", "F1": "{:.2f}"}), width="stretch")
        c1, c2, c3 = st.columns(3)
        c1.metric("Cases", len(cs)); c2.metric("Anticipation cases", len(ant)); c3.metric("Dry-down events (anticipation)", int(ant.stress72.sum()))
        st.bar_chart(cs.action.value_counts())
        st.warning("Few independent events: treat these as indicative only.")
        st.dataframe(cs.drop(columns=[c for c in cs.columns if c.startswith("fc")]).tail(200), width="stretch")

    # ---------------------------------------------------------------- Explain
    with tabs[5]:
        import joblib
        fmod = joblib.load(ARTIFACTS / "forecast_models.joblib"); rmod = joblib.load(ARTIFACTS / "risk_models.joblib")
        kind = st.radio("Model", ["Forecast (median, change)", "Dry-risk classifier"], horizontal=True)
        Hx = st.selectbox("Horizon", [24, 48, 72], key="ex_h")
        if kind.startswith("Forecast"):
            m, cols = fmod["horizons"][Hx]["mean"], fmod["cols"]
        else:
            m, cols = rmod["models"][Hx], rmod["cols"] + ["gap_to_trigger"]
        imp = pd.DataFrame({"feature": m.booster_.feature_name(), "gain": m.booster_.feature_importance("gain")})
        imp["share"] = imp.gain / imp.gain.sum()
        imp = imp.sort_values("share", ascending=False).head(25)
        st.altair_chart(alt.Chart(imp).mark_bar().encode(x="share:Q", y=alt.Y("feature:N", sort="-x"), tooltip=["gain", "share"]).properties(height=520), width="stretch")
        trust_share = imp[imp.feature.str.startswith(("trust_", "batt", "anom"))].share.sum()
        st.caption(f"Trust indicators in the top 25: {trust_share:.1%} of gain. Importance is not causation.")

    # ---------------------------------------------------------------- Case inspector
    with tabs[6]:
        from irrigation_ml.decision import Policy
        from irrigation_ml.recommend import recommend, render
        df, feat = prepared()
        c1, c2, c3 = st.columns(3)
        day = c1.date_input("Date", pd.Timestamp("2025-05-05").date(), min_value=feat.timestamp.min().date(), max_value=feat.timestamp.max().date())
        hour = c2.slider("Hour", 0, 23, 6)
        cf = c3.select_slider("c_fn", [1, 2, 3, 5, 10, 20], value=5, key="ci_c")
        recs = recommend(f"{day}T{hour:02d}:00", Policy(c_fn=float(cf)), "sprinkler", (df, feat))
        st.code(render(recs))
        st.json(recs, expanded=False)
        st.subheader("Feature row at issue time")
        fld = st.selectbox("Field", [r["field_id"] for r in recs], key="ci_f")
        row = feat[(feat.field_id == fld) & (feat.timestamp <= pd.Timestamp(f"{day}T{hour:02d}:00"))].iloc[[-1]].T
        row.columns = ["value"]; row["value"] = row["value"].astype(str)
        st.dataframe(row, height=400, width="stretch")

    # ---------------------------------------------------------------- Pipeline
    with tabs[7]:
        st.subheader("Artifacts")
        af = pd.DataFrame([dict(file=p.name, size_kb=round(p.stat().st_size / 1024, 1), modified=datetime.fromtimestamp(p.stat().st_mtime)) for p in sorted(ARTIFACTS.iterdir())])
        st.dataframe(af, hide_index=True, width="stretch")
        st.subheader("Input tables")
        st.dataframe(pd.DataFrame([dict(file=p.name, size_kb=round(p.stat().st_size / 1024, 1), modified=datetime.fromtimestamp(p.stat().st_mtime)) for p in sorted(PROCESSED.glob("*"))]), hide_index=True, width="stretch")
        meta = PROCESSED / "pipeline_meta.json"
        if meta.exists():
            mj = json.loads(meta.read_text())
            st.write("DE trust features (hourly):", mj.get("trust_features_hourly", "missing: re-export with the patched notebook"))
            with st.expander("DE assumptions"):
                st.write(mj.get("assumptions"))
        c1, c2 = st.columns(2)
        if c1.button("Run unit tests"):
            out = subprocess.run([sys.executable, "-m", "unittest", "tests.test_irrigation_ml", "-v"], cwd=ROOT, capture_output=True, text=True)
            st.code(out.stderr[-3000:])
        if c2.button("Retrain everything (~20 s)"):
            with st.spinner("Training..."):
                out = subprocess.run([sys.executable, "-m", "irrigation_ml.pipeline"], cwd=ROOT, capture_output=True, text=True, env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")})
            st.code((out.stdout + out.stderr)[-2000:]); st.cache_data.clear(); st.cache_resource.clear()
