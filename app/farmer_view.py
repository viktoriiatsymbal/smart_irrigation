"""Farmer view: render() is called from app.py."""
import altair as alt
import pandas as pd
import streamlit as st

from common import artifacts_ready, prepared, raw_sensor, recommendations

T = {
    "en": dict(
        title="Smart Irrigation", prev_day="← Previous day", next_day="Next day →", field="Field", date="Date (demo on historical data)", system="Irrigation system",
        sprinkler="Sprinkler", drip="Drip", caution="Caution level", cautious="Very cautious", balanced="Balanced", saving="Save water",
        caution_help="Cautious: irrigate earlier, minimal risk of missing a dry spell. Saving: fewer irrigations, higher risk of missing one.",
        demo="Demo mode: showing data as of",
        irrigate="IRRIGATE TODAY", postpone="POSTPONE — rain is coming", no_action="NO IRRIGATION NEEDED", check_sensor="CHECK THE SENSOR",
        why_irrigate="The soil is already dry or will be within a day.",
        why_postpone="Rain is forecast and the moisture forecast shows the soil staying moist enough.",
        why_no_action="The soil is unlikely to dry out within 3 days.",
        irrigate_soon_1="IRRIGATE TOMORROW", irrigate_soon_n="PLAN TO IRRIGATE IN {n} DAYS",
        why_irrigate_soon="Soil moisture is forecast to reach the irrigation level within {d} days. Irrigating {when} keeps the crop ahead of water stress.",
        why_irrigate_soon_est="Without the sensor, the weather suggests the soil will reach the irrigation level within {d} days. Irrigating {when} keeps you ahead of it.",
        when_1="by tomorrow", when_n="within {n} days", by_date="Irrigate by {date}",
        plan="3-day plan", plan_est="3-day plan without the sensor (weather-based)", plan_action="Suggested action", plan_today="Today", plan_tomorrow="Tomorrow",
        act_irrigate="Irrigate", act_irrigate_day="Irrigate (recommended day)", act_dry="Dry soil expected", act_not_yet="No irrigation yet",
        act_monitor="Check again", act_none="No irrigation needed", act_rain="Rain expected", act_postpone="Postpone (rain expected)", act_dash="—",
        why_check="No reliable advice can be given until the sensor is working again.",
        why_irrigate_est="The soil may be getting dry. This is an estimate from the last sensor reading and the weather.",
        why_no_action_est="The soil is unlikely to dry out within 3 days, judging by the last sensor reading and the weather.",
        est_line="Estimate without the sensor", sensor_title="Sensor problem", unusual_title="Unusual sensor readings",
        deficit_note="The soil is about {d:.0f} mm short. Apply {a:.0f} mm now and repeat on the following days.",
        est_outlook="Outlook without the sensor (weather-based)",
        moisture="Soil moisture", trigger="Irrigate below", amount="Suggested amount",
        amount_val="{mm:.0f} mm", amount_sub="≈ {m3:.0f} m³ per hectare", none="—", outlook="3-day outlook", chart="Soil moisture and forecast",
        day="Day", hours="h", expected="Expected moisture", risk="Risk of drying out", rain="Forecast rain (mm)",
        sensor="Sensor health", battery="Battery", last="Last reading", ok="OK", warn="needs attention",
        h_ago="{h:.0f} h ago", conf="Confidence",
        c_high="high", c_medium="medium", c_low="low",
        low_conf_probe="This field's probe barely changes, so the trigger is approximate.",
        disclaimer="This is advice, not automatic control: the irrigation decision is yours. The model was trained on two fields and has not been validated against real irrigation records.",
        coverage="Probe coverage, 24 h", now_line="Now {sm:.0f}, trigger {th:.0f}", unit="Probe reading on a 0–100 scale (higher = wetter)",
        trig_label="irrigate below this line", past="measured", fc="forecast", band="80% range",
        x_title="Date", y_title="Soil moisture (higher = wetter)", tip_measured="Measured", tip_expected="Expected",
        tip_low="Lowest likely", tip_high="Highest likely",
        chart_hist="Soil moisture, last 14 days", no_history="The sensor sent no readings in the last 14 days, so there is nothing to show.", tip_untrusted="Not trusted",
        legend_trusted="The grey line is what your sensor measured over the last 14 days (readings the system trusts).",
        legend_untrusted="Orange dots are readings the system does not trust (sensor fault or unusual values); they are not used for advice.",
        legend_forecast="The blue dashed line is the expected moisture over the next 3 days, and the blue shaded area is the range where it will most likely be (about 80% chance).",
        legend_rule="The red dotted line is the level below which irrigation is recommended."),
}
CAUTION = {"cautious": 10.0, "balanced": 5.0, "saving": 2.0}
STYLE = {"irrigate": "#f2c200", "irrigate_soon": "#1d6fb8", "postpone": "#7a5ccf", "no_action": "#2e8b57", "check_sensor": "#c0392b"}

t = T["en"]


def render():
    if not artifacts_ready():
        st.error("Models not trained yet. Run: PYTHONPATH=src python -m irrigation_ml.pipeline")
        st.stop()

    df, feat = prepared()
    with st.sidebar:
        fields = sorted(feat["field_id"].unique())
        field = st.selectbox(t["field"], fields, key="f_field")
        lo, hi = feat["timestamp"].min().date(), feat["timestamp"].max().date()
        st.session_state.setdefault("f_day", min(max(pd.Timestamp("2025-05-05").date(), lo), hi))

        def _step(delta):                      # callback: runs before the rerun, so it may set the widget's state
            st.session_state["f_day"] = min(max(st.session_state["f_day"] + pd.Timedelta(days=delta), lo), hi)

        day = st.date_input(t["date"], min_value=lo, max_value=hi, key="f_day")
        b1, b2 = st.columns(2)
        b1.button(t["prev_day"], on_click=_step, args=(-1,), disabled=day <= lo, width="stretch", key="f_prev")
        b2.button(t["next_day"], on_click=_step, args=(1,), disabled=day >= hi, width="stretch", key="f_next")
        system = st.radio(t["system"], ["sprinkler", "drip"], format_func=lambda x: t[x], horizontal=True, key="f_system")
        level = st.select_slider(t["caution"], options=list(CAUTION), value="balanced", format_func=lambda x: t[x], help=t["caution_help"], key="f_level")

    at = f"{day}T06:00"
    recs = {r["field_id"]: r for r in recommendations(at, CAUTION[level], system)}
    r = recs.get(field)
    st.title(t["title"])
    st.caption(f"{t['demo']} {day} 06:00 · {field}")
    if r is None:
        st.warning("No data for this field on the selected date."); st.stop()

    color = STYLE[r["action"]]
    fg = "#1f1f1f" if r["action"] == "irrigate" else "white"      # white text is unreadable on yellow
    est = r["mode"] == "weather_only"
    why = {"irrigate": t["why_irrigate_est" if est else "why_irrigate"], "postpone": t["why_postpone"],
           "no_action": t["why_no_action_est" if est else "why_no_action"], "check_sensor": t["why_check"]}.get(r["action"])
    title = t.get(r["action"])
    if r["action"] == "irrigate_soon":
        n = r["irrigate_in_h"] // 24
        title = t["irrigate_soon_1"] if n == 1 else t["irrigate_soon_n"].format(n=n)
        when = t["when_1"] if n == 1 else t["when_n"].format(n=n)
        why = t["why_irrigate_soon_est" if est else "why_irrigate_soon"].format(d=r["dry_expected_h"] // 24, when=when)
        why += f"<br><b>{t['by_date'].format(date=pd.Timestamp(r['irrigate_by']).strftime('%A %d %b'))}</b>"
    foot = f"{t['est_line']} · {t['conf'].lower()}: {t['c_low']}" if est else f"{t['conf']}: {t['c_' + r['confidence']]}"
    if r["sensor_issue"]:
        (st.warning if r["mode"] != "sensor" else st.info)(f"**{t['sensor_title'] if r['mode'] != 'sensor' else t['unusual_title']}.** {r['sensor_issue']}")
    st.markdown(
        f"""<div style="background:{color};color:{fg};padding:26px 24px;border-radius:16px;margin:8px 0 18px 0">
        <div style="font-size:30px;font-weight:700;margin-top:6px">{title}</div>
        <div style="font-size:17px;opacity:.95;margin-top:6px">{why}</div>
        <div style="font-size:13px;opacity:.85;margin-top:10px">{foot}</div></div>""",
        unsafe_allow_html=True)

    sm, trig = r["soil_moisture"], r["trigger"]
    if r["mode"] != "sensor":
        sm = None                          # an unusable reading is not shown as if it were reliable
    c1, c2, c3 = st.columns(3)
    c1.metric(t["moisture"], f"{sm:.0f}" if sm is not None else t["none"], help=t["unit"])
    c2.metric(t["trigger"], f"{trig:.0f}", help=t["unit"])
    if r["amount_mm"]:
        c3.metric(t["amount"], t["amount_val"].format(mm=r["amount_mm"]), delta=t["amount_sub"].format(m3=r["amount_mm"] * 10),
                  delta_color="off", delta_arrow="off", help="1 mm = 1 L/m² = 10 m³/ha")
    else:
        c3.metric(t["amount"], t["none"])
    if r.get("capped"):
        st.caption(t["deficit_note"].format(d=r["deficit_mm"], a=r["amount_mm"]))
    if r["confidence"] == "low" and r["mode"] == "sensor" and not r["sensor_issue"]:
        st.info(t["low_conf_probe"])

    issued = pd.Timestamp(r["issued_at"])
    win = (feat.field_id == field) & (feat.timestamp > issued - pd.Timedelta(days=14)) & (feat.timestamp <= issued)
    trusted = feat.loc[win, ["timestamp", "sm0"]].rename(columns={"sm0": "mid"}).dropna()
    rw = raw_sensor()
    rw = rw[(rw.field_id == field) & (rw.timestamp > issued - pd.Timedelta(days=14)) & (rw.timestamp <= issued)]
    rw = rw.merge(feat.loc[win, ["timestamp", "sm0"]], on="timestamp", how="left")
    untrusted = rw[rw.soil_moisture_sensor.notna() & rw.sm0.isna()].rename(columns={"soil_moisture_sensor": "mid"})[["timestamp", "mid"]]
    has_fc = bool(r["forecast"])
    if has_fc:
        fc = pd.DataFrame([dict(timestamp=issued + pd.Timedelta(hours=H), mid=v[1], lo=v[0], hi=v[2]) for H, v in r["forecast"].items()])
        fcl = pd.concat([pd.DataFrame([dict(timestamp=issued, mid=sm, lo=sm, hi=sm)]), fc])
    if len(trusted) or len(untrusted) or has_fc:
        fc_color = "#1d6fb8"
        ys = [trig, *trusted.mid, *untrusted.mid] + ([*fc.lo, *fc.hi] if has_fc else [])
        y_dom = [max(0, min(ys) - 5), 102]
        x_enc = alt.X("timestamp:T", title=t["x_title"], axis=alt.Axis(format="%d %b"))
        y_enc = alt.Y("mid:Q", title=t["y_title"], scale=alt.Scale(domain=y_dom))
        tip_date = alt.Tooltip("timestamp:T", title=t["x_title"], format="%d %b %H:%M")
        layers = []
        if has_fc:
            layers.append(alt.Chart(fcl).mark_area(opacity=.25, color=fc_color).encode(x=x_enc, y=alt.Y("lo:Q", scale=alt.Scale(domain=y_dom)), y2="hi:Q"))
        if len(trusted):
            layers.append(alt.Chart(trusted).mark_line(color="#888").encode(x=x_enc, y=y_enc, tooltip=[tip_date, alt.Tooltip("mid:Q", title=t["tip_measured"], format=".0f")]))
        if len(untrusted):
            layers.append(alt.Chart(untrusted).mark_point(color="#e08a1e", filled=True, size=22, opacity=.8).encode(
                x=x_enc, y=y_enc, tooltip=[tip_date, alt.Tooltip("mid:Q", title=t["tip_untrusted"], format=".0f")]))
        if has_fc:
            layers.append(alt.Chart(fcl).mark_line(color=fc_color, strokeDash=[5, 3], point=True).encode(
                x=x_enc, y=y_enc, tooltip=[tip_date, alt.Tooltip("mid:Q", title=t["tip_expected"], format=".0f"),
                                           alt.Tooltip("lo:Q", title=t["tip_low"], format=".0f"), alt.Tooltip("hi:Q", title=t["tip_high"], format=".0f")]))
        x0 = min([d.timestamp.min() for d in (trusted, untrusted) if len(d)] + [issued])
        layers.append(alt.Chart(pd.DataFrame({"y": [trig]})).mark_rule(color="#c0392b", strokeDash=[2, 2]).encode(y="y:Q"))
        layers.append(alt.Chart(pd.DataFrame({"y": [trig], "l": [t["trig_label"]], "x": [x0]})).mark_text(align="left", dy=-6, color="#c0392b").encode(x="x:T", y="y:Q", text="l:N"))
        st.subheader(t["chart"] if has_fc else t["chart_hist"])
        st.altair_chart(alt.layer(*layers).properties(height=280), width="stretch")
        legend = ([t["legend_trusted"]] if len(trusted) else []) + ([t["legend_untrusted"]] if len(untrusted) else []) \
            + ([t["legend_forecast"]] if has_fc else []) + [t["legend_rule"]]
        st.caption("**How to read this:** " + " ".join(legend))
    else:
        st.subheader(t["chart_hist"]); st.caption(t["no_history"])

    if r["p_dry"]:
        st.subheader(t["plan_est"] if r["mode"] == "weather_only" else t["plan"])
        rows = []
        for H in (0, 24, 48, 72):
            day = issued + pd.Timedelta(hours=H)
            lbl = t["plan_today"] if H == 0 else (t["plan_tomorrow"] if H == 24 else "")
            a_ = r["action"]
            if a_ == "irrigate_soon":
                act = t["act_irrigate_day"] if H == r["irrigate_in_h"] else t["act_dry"] if H == r["dry_expected_h"] else t["act_not_yet"] if H < r["irrigate_in_h"] else t["act_monitor"]
            elif a_ == "irrigate":
                act = t["act_irrigate"] if H == 0 else t["act_monitor"]
            elif a_ == "postpone":
                act = t["act_postpone"] if H == 0 or r["rain_forecast_mm"].get(H, 0) >= 5 else t["act_dash"]
            else:
                act = t["act_none"]
            row = {t["day"]: (lbl + ", " if lbl else "") + day.strftime("%a %d %b"), t["plan_action"]: act}
            if H and r["forecast"]:
                v = r["forecast"][H]; row[t["expected"]] = f"{v[1]:.0f} ({v[0]:.0f}–{v[2]:.0f})"
            elif r["forecast"]:
                row[t["expected"]] = f"{sm:.0f}" if sm is not None else t["none"]
            row[t["risk"]] = f"{r['p_dry'][H]:.0%}" if H else t["none"]
            row[t["rain"]] = f"{r['rain_forecast_mm'][H]:.1f}" if H else t["none"]
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.subheader(t["sensor"])
    row = feat[(feat.field_id == field) & (feat.timestamp <= pd.Timestamp(r["issued_at"]))].iloc[-1]
    age = (pd.Timestamp(at) - pd.Timestamp(r["issued_at"])).total_seconds() / 3600
    batt = row["batt_min_24"]
    b_ok = pd.isna(batt) or batt >= 3.8
    s1, s2, s3 = st.columns(3)
    s1.metric(t["battery"], f"{batt:.2f} V" if pd.notna(batt) else t["none"], delta=t["ok"] if b_ok else t["warn"], delta_color="normal" if b_ok else "inverse")
    s2.metric(t["last"], t["h_ago"].format(h=age), delta=t["ok"] if age <= 6 else t["warn"], delta_color="normal" if age <= 6 else "inverse")
    s3.metric(t["coverage"], f"{row['obs_share_24']:.0%}")
    st.caption(t["disclaimer"])
