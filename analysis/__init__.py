"""Standalone analysis & statistics tooling for Active Learning experiments.

No Streamlit dependency — pure file reading + numpy/pandas/scipy/matplotlib so it can run
on the cluster or locally and produce thesis-ready figures and tables from experiments/.

Entry point: `python -m analysis.run --exp-dir experiments --out analysis_out`
"""
