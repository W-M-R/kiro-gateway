#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Kiro Gateway - Quota Status CLI.

Queries all accounts in kiro-cli-db-file/ and displays their remaining
quota in a formatted table. Can also poll continuously.

Usage:
    python scripts/quota_status.py              # One-shot query
    python scripts/quota_status.py --watch      # Continuous refresh (5s)
    python scripts/quota_status.py --json       # JSON output
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from kiro.auth import KiroAuthManager
from kiro.config import get_httpx_verify_config
from kiro.quota import QuotaInfo, query_quota


def _format_reset(epoch: float) -> str:
    """Format reset timestamp as readable date."""
    if epoch <= 0:
        return "—"
    dt = datetime.fromtimestamp(epoch, tz=timezone.utc)
    return dt.strftime("%m-%d %H:%M UTC")


def _color(text: str, code: str) -> str:
    """Wrap text in ANSI color code."""
    return f"\033[{code}m{text}\033[0m"


def render_table(quota_list: List[QuotaInfo]) -> str:
    """Render quota list as a formatted ASCII table."""
    if not quota_list:
        return "No accounts found."

    # Sort: exhausted last, then by total_remaining descending
    sorted_list = sorted(quota_list, key=lambda q: (q.is_exhausted, -q.total_remaining))

    # Column widths
    owner_w = max(len("Owner"), max(len(q.owner) for q in sorted_list))
    sub_w = max(len("Subscription"), max(len(q.subscription) for q in sorted_list))

    # Header
    header = (
        f"{'Owner':<{owner_w}}  "
        f"{'Subscription':<{sub_w}}  "
        f"{'Free Used/Limit':>16}  "
        f"{'Overage Used/Cap':>17}  "
        f"{'Remaining':>10}  "
        f"{'Reset':>14}  "
        f"{'Status':>8}"
    )
    separator = "─" * len(header)

    lines = [separator, header, separator]

    for q in sorted_list:
        free_str = f"{q.current_usage}/{q.usage_limit}"
        overage_str = f"{q.current_overages}/{q.overage_cap}"

        if q.is_exhausted:
            remaining_str = _color("0", "91")  # red
            status_str = _color("DEAD", "91")
        elif q.total_remaining < 1000:
            remaining_str = _color(str(q.total_remaining), "93")  # yellow
            status_str = _color("LOW", "93")
        else:
            remaining_str = _color(str(q.total_remaining), "92")  # green
            status_str = _color("OK", "92")

        row = (
            f"{q.owner:<{owner_w}}  "
            f"{q.subscription:<{sub_w}}  "
            f"{free_str:>16}  "
            f"{overage_str:>17}  "
            f"{remaining_str:>10}  "
            f"{_format_reset(q.next_reset_epoch):>14}  "
            f"{status_str:>8}"
        )

        if q.last_error:
            row += f"\n  └─ Error: {q.last_error}"

        lines.append(row)

    lines.append(separator)

    # Summary
    total = len(sorted_list)
    available = len([q for q in sorted_list if not q.is_exhausted])
    exhausted = total - available
    total_remaining = sum(q.total_remaining for q in sorted_list)

    summary = (
        f"Accounts: {total} total, {_color(str(available), '92')} available, "
        f"{_color(str(exhausted), '91')} exhausted  |  "
        f"Total remaining: {_color(str(total_remaining), '96')}"
    )
    lines.append(summary)

    return "\n".join(lines)


async def query_all(db_dir: Path) -> List[QuotaInfo]:
    """Query quota for all accounts in db_dir."""
    results: List[QuotaInfo] = []
    owners = sorted([p.name for p in db_dir.iterdir() if p.is_dir()])

    for owner in owners:
        db_path = db_dir / owner / "data.sqlite3"
        if not db_path.exists():
            continue

        try:
            auth_manager = KiroAuthManager(
                sqlite_db=str(db_path),
                region="us-east-1",
            )
            quota_info = await query_quota(auth_manager, owner, str(db_path))
            results.append(quota_info)
        except Exception as e:
            now = time.time()
            results.append(QuotaInfo(
                owner=owner,
                account_id=str(db_path),
                is_exhausted=True,
                last_updated=now,
                last_error=str(e),
            ))

    return results


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Kiro Gateway - Quota Status CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--db-dir",
        type=str,
        default="kiro-cli-db-file",
        help="Directory containing owner subdirectories with data.sqlite3 (default: kiro-cli-db-file)",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuously refresh every 5 seconds",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of table",
    )
    args = parser.parse_args()

    db_dir = Path(args.db_dir)
    if not db_dir.is_absolute():
        db_dir = PROJECT_ROOT / db_dir

    if not db_dir.exists():
        print(f"Error: DB directory not found: {db_dir}", file=sys.stderr)
        return 1

    while True:
        quota_list = await query_all(db_dir)

        if args.json:
            print(json.dumps([q.to_dict() for q in quota_list], indent=2, ensure_ascii=False))
        else:
            if args.watch:
                print("\033[2J\033[H", end="")  # Clear screen
            print(f"\nKiro Gateway Quota Status — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            print(render_table(quota_list))

        if not args.watch:
            break

        await asyncio.sleep(5)

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
