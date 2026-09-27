"""Capture dashboard screenshots for the README (needs the dashboard running on --port).

Run: streamlit run dashboard/app.py --server.port 8599   then   python -m scripts.screenshots
Uses the locally installed Microsoft Edge through Playwright (no browser download).
"""
import sys

from playwright.sync_api import sync_playwright

from app.config import ROOT

PORT = sys.argv[1] if len(sys.argv) > 1 else "8599"
OUT = ROOT / "docs" / "images"
SHOTS = [("dash_overview.png", "page=0", 900), ("dash_issue.png", "page=1", 1250),
         ("dash_trace.png", "page=2&sid=SES-0009", 1080), ("dash_eval.png", "page=3", 1350)]

with sync_playwright() as p:
    browser = p.chromium.launch(channel="msedge", headless=True)
    for name, query, height in SHOTS:
        page = browser.new_page(viewport={"width": 1500, "height": height}, device_scale_factor=1)
        page.goto(f"http://localhost:{PORT}/?{query}")
        page.wait_for_selector("h1", timeout=60000)
        page.wait_for_timeout(4000)  # let dataframes and markdown finish rendering
        page.screenshot(path=str(OUT / name))
        print("saved", name)
        page.close()
    browser.close()
