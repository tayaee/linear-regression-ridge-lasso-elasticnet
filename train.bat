@echo off
set UV_CACHE_DIR=%CD%\..\..\..\..\..\.uv-cache
uv run python train.py
