
  

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


