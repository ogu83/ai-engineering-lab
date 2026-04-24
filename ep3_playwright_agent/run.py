"""CLI entry point.

Usage:
    python -m ep3_playwright_agent.run URL "goal"
    python -m ep3_playwright_agent.run URL "goal" --max-iterations 15 --verbose
"""
import argparse
import asyncio

from ep3_playwright_agent.agent.loop import run_agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Browser automation agent")
    parser.add_argument("url", help="Starting URL for the agent")
    parser.add_argument("goal", help="Natural-language goal for the agent to complete")
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--verbose", action="store_true", help="Print the full action trace")
    args = parser.parse_args()

    result, history = asyncio.run(run_agent(args.url, args.goal, args.max_iterations))

    if args.verbose:
        for i, action in enumerate(history, 1):
            line = f"[{i}] {action.action:<12} → {action.target}"
            if action.value:
                line += f" = {action.value!r}"
            print(line)
            print(f"     reason: {action.reason}")
        print()

    print(f"Result: {result}")


if __name__ == "__main__":
    main()
