@echo off
set UV_CACHE_DIR=%CD%\..\..\..\..\..\.uv-cache
uv run streamlit run web.py --server.port 80
