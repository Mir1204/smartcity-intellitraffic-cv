import streamlit as st
import json
import os
import sys
import time
import numpy as np
import cv2
import plotly.graph_objects as go

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

st.set_page_config(page_title="AI Traffic Signal Optimizer", layout="wide")
st.title("AI-Based Traffic Signal Optimization System")

tab1, tab2, tab3 = st.tabs([
    "Vehicle Detection (YOLO)",
    "Signal Time Prediction (XGBoost)",
    "Simulation Results (SUMO)",
])

# ─────────────────────────────────────────────────────
# TAB 1: YOLO Live Detection
# ─────────────────────────────────────────────────────
with tab1:
    st.header("Upload Traffic Video — Live Detection")
    st.write(
        "Upload a video and watch YOLOv8 detect vehicles **frame by frame** in real time. "
        "Unique vehicle counts update live using centroid tracking."
    )

    uploaded = st.file_uploader(
        "Choose a video file...",
        type=["mp4", "avi", "mov", "flv", "mkv"],
    )

    # ── Speed controls ────────────────────────────────────────────────────
    ctrl_col1, ctrl_col2 = st.columns(2)

    with ctrl_col1:
        frame_skip = st.select_slider(
            "Frame sampling (process every Nth frame)",
            options=[3, 5, 8, 10, 15],
            value=5,
            format_func=lambda x: {
                3:  "Every 3rd  — balanced",
                5:  "Every 5th  — fast",
                8:  "Every 8th  — faster ",
                10: "Every 10th — very fast",
                15: "Every 15th — maximum speed ",
            }[x],
        )

    with ctrl_col2:
        max_frames = st.select_slider(
            "Max frames to process",
            options=[100, 200, 300, 500, 999999],
            value=300,
            format_func=lambda x: "All frames" if x == 999999 else f"First {x} frames",
        )
        if max_frames != 999999:
            st.caption(f"⚡ Only the first {max_frames} frames will be processed — enough for accurate counting.")

    if uploaded is not None:
        from layer1_yolo.detector import process_video_live
        import tempfile

        output_dir = os.path.join(PROJECT_ROOT, "layer4_dashboard", "processed_videos")
        os.makedirs(output_dir, exist_ok=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        output_video_path = os.path.join(output_dir, f"detected_{int(time.time())}.mp4")

        # ── Live preview layout ───────────────────────────────────────
        st.subheader("Live Detection Feed")
        live_left, live_right = st.columns([3, 2])

        with live_left:
            frame_placeholder = st.empty()
            progress_bar      = st.progress(0.0)
            progress_text     = st.empty()

        with live_right:
            st.markdown("**Current frame — vehicles on screen**")
            m_cars_now  = st.empty()
            m_buses_now = st.empty()
            m_bikes_now = st.empty()

            st.divider()

            st.markdown("**Cumulative unique vehicles tracked**")
            m_cars_total  = st.empty()
            m_buses_total = st.empty()
            m_bikes_total = st.empty()

            st.divider()
            elapsed_text = st.empty()

        # Initial zeros
        m_cars_now.metric("Cars on screen",   0)
        m_buses_now.metric("Buses on screen", 0)
        m_bikes_now.metric("Bikes on screen", 0)
        m_cars_total.metric("Total unique cars",   0)
        m_buses_total.metric("Total unique buses", 0)
        m_bikes_total.metric("Total unique bikes", 0)

        # ── Run generator ─────────────────────────────────────────────
        final_result  = None
        t_start       = time.time()
        DISPLAY_EVERY = frame_skip  # refresh display once per processed frame

        try:
            for i, data in enumerate(
                process_video_live(
                    tmp_path,
                    output_path=output_video_path,
                    confidence_threshold=0.5,
                    frame_skip=frame_skip,
                    max_frames=max_frames,       # ← cap passed to generator
                )
            ):
                if i % DISPLAY_EVERY == 0 or data["done"]:
                    rgb = cv2.cvtColor(data["annotated"], cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(rgb, width='stretch')

                    progress_bar.progress(min(data["progress"], 1.0))
                    progress_text.caption(
                        f"Frame {data['frame_number']} / "
                        f"{min(max_frames, data['total_frames'])}  |  "
                        f"{data['timestamp']:.1f}s in video"
                    )

                    cur     = data["current"]
                    unq     = data["unique"]
                    elapsed = time.time() - t_start

                    m_cars_now.metric("Cars on screen",   cur["car_count"])
                    m_buses_now.metric("Buses on screen", cur["bus_truck_count"])
                    m_bikes_now.metric("Bikes on screen", cur["bike_count"])

                    m_cars_total.metric("Total unique cars",   unq["car_count"])
                    m_buses_total.metric("Total unique buses", unq["bus_truck_count"])
                    m_bikes_total.metric("Total unique bikes", unq["bike_count"])

                    elapsed_text.caption(f"Processing time: {elapsed:.1f}s")

                if data["done"]:
                    final_result = data
                    break

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        # ── Post-processing results ───────────────────────────────────
        if final_result:
            progress_bar.progress(1.0)
            progress_text.caption("Processing complete!")

            unique_totals = final_result["unique_totals"]
            window_counts = final_result["window_counts"]
            fps           = final_result["fps"]
            total_frames  = final_result["total_frames"]
            duration      = total_frames / fps if fps else 0

            st.session_state["counts"] = {
                "car_count":       unique_totals["car_count"],
                "bus_truck_count": unique_totals["bus_truck_count"],
                "bike_count":      unique_totals["bike_count"],
            }

            st.divider()

            st.subheader("Final Count Summary")
            s1, s2, s3, s4, s5 = st.columns(5)
            s1.metric("Frames Processed", final_result["frame_number"])
            s2.metric("Duration",         f"{duration:.1f}s")
            s3.metric("FPS",              fps)
            s4.metric("1-s Windows",      len(window_counts))
            s5.metric("Unique Vehicles",
                      unique_totals["car_count"] +
                      unique_totals["bus_truck_count"] +
                      unique_totals["bike_count"])

            u1, u2, u3 = st.columns(3)
            u1.metric(" Unique Cars",      unique_totals["car_count"])
            u2.metric(" Unique Bus/Truck", unique_totals["bus_truck_count"])
            u3.metric(" Unique Bikes",     unique_totals["bike_count"])

            st.caption(
                "Each vehicle is assigned one tracking ID. "
                "These numbers count distinct vehicles across the processed segment."
            )

            # Timeline chart
            if window_counts:
                st.subheader("Vehicles on Screen — Peak per 1-Second Window")
                windows  = [w["window"]         for w in window_counts]
                cars_tl  = [w["car_count"]       for w in window_counts]
                buses_tl = [w["bus_truck_count"] for w in window_counts]
                bikes_tl = [w["bike_count"]      for w in window_counts]

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=windows, y=cars_tl, mode="lines+markers", name="Cars",
                    line=dict(color="#3b82f6", width=2),
                    fill="tozeroy", fillcolor="rgba(59,130,246,0.08)",
                ))
                fig.add_trace(go.Scatter(
                    x=windows, y=buses_tl, mode="lines+markers", name="Buses/Trucks",
                    line=dict(color="#ef4444", width=2),
                    fill="tozeroy", fillcolor="rgba(239,68,68,0.08)",
                ))
                fig.add_trace(go.Scatter(
                    x=windows, y=bikes_tl, mode="lines+markers", name="Bikes",
                    line=dict(color="#f97316", width=2),
                    fill="tozeroy", fillcolor="rgba(249,115,22,0.08)",
                ))
                fig.update_layout(
                    xaxis_title="Window (1s each)",
                    yaxis_title="Vehicles on screen",
                    hovermode="x unified",
                    height=320,
                    margin=dict(t=20, b=40, l=40, r=20),
                    legend=dict(orientation="h", y=1.1),
                )
                st.plotly_chart(fig, width='stretch')

            # Annotated video playback
            st.subheader("Annotated Video Playback")
            if os.path.exists(output_video_path):
                st.video(output_video_path)
                with open(output_video_path, "rb") as vf:
                    st.download_button(
                        label=" Download Processed Video",
                        data=vf,
                        file_name=f"traffic_detected_{int(time.time())}.mp4",
                        mime="video/mp4",
                        width='stretch',
                    )
            else:
                st.warning("Annotated video not saved — check file permissions.")

            frame_logs = final_result.get("frame_logs", [])
            if frame_logs:
                with st.expander("View frame-by-frame details (first 30 frames)"):
                    st.table([
                        {
                            "Frame":      f["frame"],
                            "Time":       f"{f['timestamp']:.2f}s",
                            "Cars":       f["car_count"],
                            "Buses":      f["bus_truck_count"],
                            "Bikes":      f["bike_count"],
                            "Detections": f["detections"],
                        }
                        for f in frame_logs[:30]
                    ])

            st.success(
                "Done! Head to **Signal Time Prediction** — "
                "counts have been auto-filled."
            )


# ─────────────────────────────────────────────────────
# TAB 2: XGBoost Prediction
# ─────────────────────────────────────────────────────
with tab2:
    st.header("Predict Optimal Green Signal Time")

    counts = st.session_state.get(
        "counts", {"car_count": 5, "bus_truck_count": 2, "bike_count": 3}
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        car  = st.slider("Car Count",       0, 200, counts["car_count"])
        bus  = st.slider("Bus/Truck Count", 0, 100, counts["bus_truck_count"])
        bike = st.slider("Bike Count",      0, 250, counts["bike_count"])
        rain = st.toggle("Rain Condition",  value=False)
    with col2:
        st.info(
            "Upload a video in **Vehicle Detection** to auto-fill these "
            "sliders with unique vehicle counts."
        )

    if st.button("Predict Green Time", type="primary"):
        from layer2_ml.predict import predict_green_time, predict_green_time_class

        green  = predict_green_time(car, bus, bike, int(rain))
        detail = predict_green_time_class(car, bus, bike, int(rain))

        col_main = st.columns([1, 2, 1])
        with col_main[1]:
            st.markdown(
                f"""<div style="text-align:center;padding:30px;
                background:linear-gradient(135deg,#4CAF50 0%,#45a049 100%);
                border-radius:15px;box-shadow:0 8px 16px rgba(0,0,0,.2);">
                    <h1 style="color:white;margin:0;font-size:72px;">{green}</h1>
                    <p style="color:white;margin:10px 0 0;font-size:24px;font-weight:bold;">SECONDS</p>
                    <p style="color:rgba(255,255,255,.9);margin:10px 0 0;font-size:16px;">Recommended Green Signal Time</p>
                    <p style="color:rgba(255,255,255,.8);margin:5px 0 0;font-size:14px;">Confidence: {detail['confidence']:.1%}</p>
                </div>""",
                unsafe_allow_html=True,
            )

        st.subheader("Signal State Preview")
        s1, s2 = st.columns(2)
        with s1:
            st.markdown(
                f"""<div style="text-align:center;">
                <div style="width:100px;height:100px;border-radius:50%;
                background:green;margin:auto;box-shadow:0 0 30px green;"></div>
                <p style="margin-top:15px;font-weight:bold;font-size:18px;">Active Phase<br>
                <span style="font-size:32px;color:green;">{green}s</span></p></div>""",
                unsafe_allow_html=True,
            )
        with s2:
            st.markdown(
                """<div style="text-align:center;">
                <div style="width:100px;height:100px;border-radius:50%;
                background:red;margin:auto;box-shadow:0 0 30px red;"></div>
                <p style="margin-top:15px;font-weight:bold;font-size:18px;">Waiting Phase</p></div>""",
                unsafe_allow_html=True,
            )

        st.subheader("Traffic Conditions")
        ic = st.columns(4)
        ic[0].metric("Cars",      car)
        ic[1].metric("Bus/Truck", bus)
        ic[2].metric("Bikes",     bike)
        ic[3].metric("Rain",      "Yes" if rain else "No")

        st.subheader("Model Confidence Breakdown")
        prob_cols = st.columns(4)
        for idx, (label, value) in enumerate(detail["probabilities"].items()):
            time_val = label.replace("class_", "").replace("s", "")
            prob_cols[idx].metric(f"{time_val}s", f"{value:.1%}")


# ─────────────────────────────────────────────────────
# TAB 3: SUMO Simulation Results
# ─────────────────────────────────────────────────────
with tab3:
    st.header("SUMO Simulation: Fixed vs Adaptive")

    results_dir = os.path.join(PROJECT_ROOT, "layer3_sumo", "results")
    comp_path   = os.path.join(results_dir, "comparison.json")

    if os.path.exists(comp_path):
        with open(comp_path) as f:
            comp = json.load(f)

        fixed    = comp["fixed"]
        adaptive = comp["adaptive"]

        st.subheader("Performance Comparison")
        m1, m2, m3 = st.columns(3)
        m1.metric("Avg Waiting Time",  f"{adaptive['avg_waiting_time']}s",
                  f"-{comp['improvement_wait_pct']}%")
        m2.metric("Avg Queue Length",  f"{adaptive['avg_queue_length']}",
                  f"-{comp['improvement_queue_pct']}%")
        m3.metric("Throughput",        f"{adaptive['total_arrived']} vehicles",
                  f"+{adaptive['total_arrived'] - fixed['total_arrived']}")

        fig = go.Figure(data=[
            go.Bar(name="Fixed Time",    x=["Avg Wait (s)", "Avg Queue"],
                   y=[fixed["avg_waiting_time"],    fixed["avg_queue_length"]],
                   marker_color="indianred"),
            go.Bar(name="Adaptive (ML)", x=["Avg Wait (s)", "Avg Queue"],
                   y=[adaptive["avg_waiting_time"], adaptive["avg_queue_length"]],
                   marker_color="seagreen"),
        ])
        fig.update_layout(barmode="group", title="Fixed vs Adaptive Signal Control",
                          yaxis_title="Value")
        st.plotly_chart(fig, width='stretch')

        fig2 = go.Figure(data=[go.Bar(
            x=["Fixed Time", "Adaptive (ML)"],
            y=[fixed["total_arrived"], adaptive["total_arrived"]],
            marker_color=["indianred", "seagreen"],
            text=[fixed["total_arrived"], adaptive["total_arrived"]],
            textposition="auto",
        )])
        fig2.update_layout(title="Vehicle Throughput (600s Simulation)",
                           yaxis_title="Vehicles Completed")
        st.plotly_chart(fig2, width='stretch')

    else:
        st.warning(
            "No simulation results found. "
            "Run `python -m layer3_sumo.compare` from the project root first."
        )

    st.divider()
    if st.button("Run Simulation Now (takes ~30s)"):
        with st.spinner("Running both SUMO simulations..."):
            from layer3_sumo.compare import compare
            compare(rain=0)
        st.rerun()
