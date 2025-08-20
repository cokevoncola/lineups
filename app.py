# app.py
# Streamlit UI for MLB SKYNET — Coladyne Systems

from __future__ import annotations
import os, io, sys, tempfile
from pathlib import Path
from datetime import date
import pandas as pd
import streamlit as st

# --- Path guard so imports from src/ always work ---
try:
    from src.pathguard import ensure_project_paths
    ensure_project_paths(__file__)
except Exception:
    pass

# --- Robust debug panel import with diagnostics ---
try:
    from src.debug_panel import manifest_status
    DEBUG_PANEL_OK = True
    DEBUG_IMPORT_ERR = None
except Exception as e:
    DEBUG_PANEL_OK = False
    DEBUG_IMPORT_ERR = e

def _debug_import_diag():
    st.write({
        "cwd": os.getcwd(),
        "__file__": str(Path(__file__).resolve()),
        "has_src_dir": Path("src").exists(),
        "src_list": sorted(os.listdir("src")) if Path("src").exists() else "src missing",
        "sys.path_sample": sys.path[:8],
        "error": str(DEBUG_IMPORT_ERR) if not DEBUG_PANEL_OK else None,
    })

# --- Optional imports ---
try:
    from src.picks import make_picks, make_picks_smart_mc
except Exception:
    make_picks = None
    make_picks_smart_mc = None

try:
    from src.smart_csv_processor import process_smart_csv
except Exception:
    process_smart_csv = None

try:
    from src.pick_selector import compute_edges_with_optional_odds, rank_strikeout_edges
except Exception:
    def compute_edges_with_optional_odds(df): return df
    def rank_strikeout_edges(df, top_n=10): return df

# fallback wrapper for lineups-aware simulation
try:
    from src.smart_monte_carlo_predictor import simulate_matchups
except Exception:
    simulate_matchups = None

st.set_page_config(page_title="MLBSKYNET — Coladyne Systems", layout="wide")

# ---------------- Sidebar controls ----------------
st.sidebar.header("Simulation Settings")
sims_per_pitcher = st.sidebar.number_input(
    "Simulations per pitcher", min_value=100, max_value=100000, step=100, value=3000
)

st.sidebar.header("Smart MC (Models) Options")
pitch_cap_mode = st.sidebar.selectbox("Pitch cap mode", options=["auto","soft","hard"], index=0)
default_pitch_cap = st.sidebar.number_input(
    "Default pitch cap (fallback)", min_value=60, max_value=140, step=1, value=95
)
hook_aggr = float(st.sidebar.slider("Hook aggressiveness", min_value=0.0, max_value=2.0, step=0.05, value=1.0))
use_trained = st.sidebar.checkbox("Use trained heads (if available)", value=True)
strict_only = st.sidebar.checkbox("Strict model-only (no heuristic overrides)", value=True)

# --- Fetcher UI (writes to data/lineups/) ---
st.sidebar.header("Lineups Fetcher (StatsAPI)")
fetch_date = st.sidebar.date_input("Fetch date", value=date.today())
if st.sidebar.button("Fetch lineups for date"):
    try:
        from tools.fetch_lineups_statsapi import write_csv_for_date
        out_path = write_csv_for_date(fetch_date.isoformat())
        st.sidebar.success(f"Saved: {out_path}")
    except Exception as e:
        st.sidebar.error(f"Fetch failed: {e}")

# ---------------- Debug blocks ----------------
with st.expander("Debug • Model Manifest & Heads", expanded=False):
    if DEBUG_PANEL_OK:
        manifest_status()
    else:
        st.error("Debug panel not available (failed to import src.debug_panel).")
        _debug_import_diag()
        st.info("Expected files: src/debug_panel.py and src/model_manifest.py")

st.title("MLBSKYNET – Coladyne Systems")
st.subheader("Generate Picks from Lineup CSV")
st.caption("Upload a lineup CSV, or pick one saved in data/lineups/. We normalize IDs, build opponent lineups, enrich with team-aware fallbacks, and run simulations.")

# --------- Source selection: uploaded OR pick existing ----------
# Existing files list (new)
lineups_dir = Path("data/lineups")
existing_files = []
if lineups_dir.exists():
    files = sorted(lineups_dir.glob("Lineups_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    existing_files = [str(p) for p in files]

sel_existing = None
if existing_files:
    sel_existing = st.selectbox(
        "Or choose an existing lineup CSV (data/lineups/)",
        options=["(none)"] + existing_files,
        index=1 if len(existing_files) else 0,
    )
    if sel_existing == "(none)":
        sel_existing = None
else:
    st.info("No files found in data/lineups/. Use the sidebar fetcher or upload manually.")

# Upload (still available)
uploaded = st.file_uploader("Upload lineup CSV (optional if you pick an existing one)", type=["csv"], accept_multiple_files=False)

def _load_and_normalize(uploaded_file, selected_path: str | None):
    """
    Reads uploaded CSV or selected path and calls process_smart_csv, returning (batters_df, lineups_df).
    Priority: uploaded_file > selected_path.
    """
    if uploaded_file is not None:
        # Read uploaded to DF
        try:
            df_raw = pd.read_csv(uploaded_file)
        except Exception:
            uploaded_file.seek(0)
            df_raw = pd.read_csv(io.BytesIO(uploaded_file.read()))
        source_label = "uploaded file"
    elif selected_path:
        df_raw = pd.read_csv(selected_path)
        source_label = f"existing file: {selected_path}"
    else:
        return pd.DataFrame(), pd.DataFrame(), None

    if process_smart_csv is None:
        return df_raw, pd.DataFrame(), source_label

    # Preferred signature (returns batters_df, lineups_df)
    try:
        batters_df, lineups_df = process_smart_csv(df_raw, return_lineups=True)
        return batters_df, lineups_df, source_label
    except TypeError:
        # Legacy fallback
        try:
            out = process_smart_csv(df_raw)
            if isinstance(out, tuple):  # (df, meta)
                out = out[0]
            return out, pd.DataFrame(), source_label
        except Exception as e:
            st.exception(e)
            return pd.DataFrame(), pd.DataFrame(), source_label

# Action buttons
colA, colB = st.columns(2)
run_basic = colA.button("Run ML Picks")
run_smart = colB.button("Run Smart MC (Models)")

# Extra debug buttons (tiny runs)
colC, colD = st.columns(2)
btn_one = colC.button("Debug one matchup (200 sims)")
btn_two = colD.button("Debug two matchups (200 sims)")

# Parameters to pass onward
sim_cfg = dict(
    simulations=sims_per_pitcher,
    cap_mode=pitch_cap_mode,
    pitch_count_cap=default_pitch_cap,
    hook_aggressiveness=hook_aggr,
    use_trained_heads=use_trained,
    strict_model_only=strict_only,
)

# ---------------- Run pipelines ----------------
if run_basic or run_smart or btn_one or btn_two:
    batters_df, lineups_df, source_label = _load_and_normalize(uploaded, sel_existing)
    if batters_df.empty:
        st.error("No lineup data loaded. Please upload a CSV or pick an existing one.")
    else:
        st.info(f"Loaded lineup rows: {len(batters_df)}  •  Source: {source_label or 'unknown'}", icon="📄")

        # ----- Input diagnostics -----
        with st.expander("Input diagnostics", expanded=False):
            st.write("batters_df:", batters_df.shape)
            st.write("lineups_df:", lineups_df.shape)
            if not batters_df.empty:
                st.write("batters_df cols:", list(batters_df.columns))
            if not lineups_df.empty:
                st.write("lineups_df cols:", list(lineups_df.columns))
                aggregated = ("lineup_mlbid" in lineups_df.columns) or ("sp_mlbid" in lineups_df.columns)
                st.write("aggregated_lineups:", aggregated)
                if not aggregated and "batting_order" in lineups_df.columns:
                    sp_rows = lineups_df[lineups_df["batting_order"].astype(str).str.upper().eq("SP")]
                    st.write("SP rows detected:", len(sp_rows))
                    st.dataframe(sp_rows.head(12), use_container_width=True)

    # ----- Optional tiny debug runs -----
    if (btn_one or btn_two) and simulate_matchups is not None and not lineups_df.empty:
        slice_games = 1 if btn_one else 2
        test_df = lineups_df.copy()
        if "game_number" in test_df.columns:
            uniq = [g for g in test_df["game_number"].dropna().unique()]
            use = set(uniq[:slice_games])
            test_df = test_df[test_df["game_number"].isin(use)]
        sims_small = dict(simulations=200, cap_mode="soft", pitch_count_cap=95,
                          hook_aggressiveness=1.0, use_trained_heads=True, strict_model_only=True)
        st.info(f"Debug run on {slice_games} matchup(s) @200 sims…")
        try:
            debug_res = simulate_matchups(batters_df, test_df, **sims_small)
            st.dataframe(debug_res, use_container_width=True, hide_index=True)
            if "error" in debug_res.columns:
                with st.expander("Simulation errors", expanded=True):
                    cols = [c for c in ["team","opponent","pitcher_name","pitcher_id","error"] if c in debug_res.columns]
                    st.dataframe(debug_res[cols], use_container_width=True)
        except Exception as e:
            st.exception(e)

    # ----- Main run -----
    if run_basic or run_smart:
        try:
            if run_smart and make_picks_smart_mc is not None:
                st.write("Running Smart MC with heads…")
                # Prefer new signature if supported
                try:
                    out = make_picks_smart_mc(batters_df, lineups_df, **sim_cfg)
                except TypeError:
                    out = make_picks_smart_mc(batters_df, **sim_cfg)
            elif run_smart and simulate_matchups is not None:
                st.write("Running Smart MC via simulate_matchups wrapper…")
                out = simulate_matchups(batters_df, lineups_df, **sim_cfg)
            elif run_basic and make_picks is not None:
                st.write("Running heuristic/legacy picks…")
                out = make_picks(batters_df, **sim_cfg)
            else:
                out = batters_df
        except Exception as e:
            st.exception(e)
            out = pd.DataFrame()

        if isinstance(out, pd.DataFrame) and not out.empty:
            # Attach edges/EV if possible
            try:
                clean = compute_edges_with_optional_odds(out)
            except Exception:
                clean = out

            st.success("Results", icon="✅")
            st.dataframe(clean, use_container_width=True, hide_index=True)

            # Show captured errors if any
            if "error" in clean.columns:
                with st.expander("Simulation errors", expanded=True):
                    cols = [c for c in ["team","opponent","pitcher_name","pitcher_id","error"] if c in clean.columns]
                    st.dataframe(clean[cols], use_container_width=True)

            # Ranked view
            try:
                ranked = rank_strikeout_edges(clean, top_n=25)
                with st.expander("Top Edges (ranked)", expanded=False):
                    st.dataframe(ranked, use_container_width=True, hide_index=True)
            except Exception:
                pass

            # Download
            try:
                csv_bytes = clean.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download results CSV",
                    data=csv_bytes,
                    file_name="mlbskynet_results.csv",
                    mime="text/csv",
                )
            except Exception:
                pass

# Footer
st.caption("© Coladyne Systems — ultra-detailed, systematic, verified.")


