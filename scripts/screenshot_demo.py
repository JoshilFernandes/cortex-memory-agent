"""One-off script: drive the live demo with Playwright and save real
screenshots into docs/screenshots/. Requires the API server already running
on localhost:8000, e.g.:

    uvicorn cortex.api:app &
    python scripts/screenshot_demo.py
"""
import time

from playwright.sync_api import sync_playwright

BASE = "http://localhost:8000"
OUT = "docs/screenshots"


def run():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 860})
        page.goto(BASE, wait_until="networkidle")
        time.sleep(0.5)

        # --- Epoch 1: first research pass, cold start, no contradictions ---
        page.click('button[data-epoch="1"]')
        page.click("#run")
        page.wait_for_function(
            "document.getElementById('status').textContent.includes('done')",
            timeout=15000,
        )
        time.sleep(1.5)  # let the graph physics settle
        page.screenshot(path=f"{OUT}/01-epoch1-graph.png")
        print("saved 01-epoch1-graph.png")

        # --- Epoch 2: re-research the same question, memory changed ---
        page.click('button[data-epoch="2"]')
        page.click("#run")
        page.wait_for_function(
            "document.getElementById('status').textContent.includes('done')",
            timeout=15000,
        )
        time.sleep(1.5)
        page.screenshot(path=f"{OUT}/02-epoch2-contradiction.png")
        print("saved 02-epoch2-contradiction.png")

        browser.close()

    # --- Temporal history for the fact that just changed ---
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 900, "height": 560})
        page.goto(
            f"{BASE}/api/history?subject=AgentBench&relation=leader", wait_until="networkidle"
        )
        page.screenshot(path=f"{OUT}/03-history-api.png")
        print("saved 03-history-api.png")
        browser.close()


if __name__ == "__main__":
    run()
