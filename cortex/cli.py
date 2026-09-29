"""Quick local CLI: run a research query against Cortex without the web UI.

    python -m cortex.cli "Who leads AgentBench?" --epoch 1
"""
from __future__ import annotations

import argparse

from cortex.engine import get_cortex


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Cortex research query from the terminal.")
    parser.add_argument("query", help="The research question to ask.")
    parser.add_argument("--epoch", type=int, default=1, help="Offline fixture epoch (1 or 2).")
    args = parser.parse_args()

    cortex = get_cortex()
    for event in cortex.run(args.query, epoch=args.epoch):
        if event["type"] == "step":
            print(f"[{event['node']}] {event['message']}")
        else:
            print("\n=== FINAL ANSWER ===")
            print(event["final_answer"])
            print(f"\n(revisions: {event['revise_count']}, contradictions: {event['contradiction_count']})")


if __name__ == "__main__":
    main()
