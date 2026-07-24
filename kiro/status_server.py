# -*- coding: utf-8 -*-

# Kiro Gateway
# https://github.com/jwadow/kiro-gateway
# Copyright (C) 2025 Jwadow
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
Independent status web server for Kiro Gateway.

Runs on a separate port (default 19998) alongside the main API server.
Provides an HTML dashboard and JSON API for monitoring account quota
and gateway health.

Routes:
    GET /           - HTML dashboard (auto-refreshing)
    GET /api/quota  - JSON: all account quota details
    GET /api/status - JSON: gateway status summary
"""

import time
from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from loguru import logger

from kiro.config import (
    APP_VERSION,
    ACCOUNT_PRIORITY,
    QUOTA_POLL_INTERVAL,
    STATUS_SERVER_PORT,
)


def create_status_app(account_manager: Any) -> FastAPI:
    """
    Create the status FastAPI application.
    
    The app holds a reference to the shared AccountManager to read
    quota and account information. It is read-only — it never
    modifies account state.
    
    Args:
        account_manager: The shared AccountManager instance.
    
    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(
        title="Kiro Gateway Status",
        description="Quota monitoring dashboard for Kiro Gateway",
        version=APP_VERSION,
    )
    
    app.state.account_manager = account_manager
    app.state.start_time = time.time()
    
    @app.get("/", response_class=HTMLResponse)
    async def dashboard() -> HTMLResponse:
        """HTML dashboard with auto-refresh showing account quota."""
        return HTMLResponse(_render_dashboard(app))
    
    @app.get("/api/quota")
    async def api_quota() -> JSONResponse:
        """JSON API: detailed quota for all accounts."""
        am = app.state.account_manager
        quota_list = am.get_all_quota_info()
        return JSONResponse({
            "accounts": [q.to_dict() for q in quota_list],
            "priority_order": ACCOUNT_PRIORITY,
            "total_accounts": len(am._accounts),
            "polled_accounts": len(quota_list),
        })
    
    @app.get("/api/status")
    async def api_status() -> JSONResponse:
        """JSON API: gateway status summary."""
        am = app.state.account_manager
        all_accounts = list(am._accounts.values())
        initialized = [a for a in all_accounts if a.auth_manager is not None]
        quota_list = am.get_all_quota_info()
        exhausted = [q.owner for q in quota_list if q.is_exhausted]
        available = [q.owner for q in quota_list if not q.is_exhausted]
        
        return JSONResponse({
            "version": APP_VERSION,
            "uptime_seconds": int(time.time() - app.state.start_time),
            "uptime_human": _format_uptime(time.time() - app.state.start_time),
            "total_accounts": len(all_accounts),
            "initialized_accounts": len(initialized),
            "polled_accounts": len(quota_list),
            "available_accounts": available,
            "exhausted_accounts": exhausted,
            "priority_order": ACCOUNT_PRIORITY,
            "quota_poll_interval": QUOTA_POLL_INTERVAL,
            "server_time": datetime.now(tz=timezone.utc).isoformat(),
        })
    
    @app.get("/health")
    async def health() -> JSONResponse:
        """Health check endpoint."""
        return JSONResponse({"status": "ok", "service": "status-server"})
    
    return app


def _format_uptime(seconds: float) -> str:
    """Format uptime in human-readable form."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds / 60)}m"
    elif seconds < 86400:
        return f"{int(seconds / 3600)}h {int((seconds % 3600) / 60)}m"
    else:
        return f"{int(seconds / 86400)}d {int((seconds % 86400) / 3600)}h"


def _format_reset_time(epoch: float) -> str:
    """Format reset timestamp as readable date string."""
    if epoch <= 0:
        return "—"
    dt = datetime.fromtimestamp(epoch, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M UTC")


def _quota_bar(used: int, limit: int, label: str, color: str) -> str:
    """Render a progress bar for quota usage."""
    if limit <= 0:
        return f'<span class="muted">{label}: —</span>'
    pct = min(100, (used / limit) * 100)
    return (
        f'<div class="bar-container" title="{label}: {used}/{limit}">'
        f'<div class="bar-fill {color}" style="width:{pct:.1f}%"></div>'
        f'<span class="bar-label">{used}/{limit}</span>'
        f'</div>'
    )


def _render_dashboard(app: FastAPI) -> str:
    """Render the full HTML dashboard page."""
    am = app.state.account_manager
    all_accounts = list(am._accounts.values())
    quota_list = am.get_all_quota_info()
    
    # Sort: priority accounts first, then by total_remaining descending
    def sort_key(q):
        is_priority = q.owner in ACCOUNT_PRIORITY
        priority_idx = ACCOUNT_PRIORITY.index(q.owner) if is_priority else len(ACCOUNT_PRIORITY)
        return (0 if is_priority else 1, priority_idx, -q.total_remaining)
    
    quota_list_sorted = sorted(quota_list, key=sort_key)
    
    uptime = _format_uptime(time.time() - app.state.start_time)
    
    # Build account rows
    rows = []
    for q in quota_list_sorted:
        is_priority = q.owner in ACCOUNT_PRIORITY
        priority_badge = '<span class="badge priority">PRIORITY</span>' if is_priority else ''
        exhausted_class = ' class="exhausted"' if q.is_exhausted else ''
        
        free_color = "green" if q.free_remaining > 0 else "red"
        overage_color = "green" if q.overage_remaining > 1000 else ("orange" if q.overage_remaining > 100 else "red")
        
        free_bar = _quota_bar(q.current_usage, q.usage_limit, "Free", free_color)
        overage_bar = _quota_bar(q.current_overages, q.overage_cap, "Overage", overage_color)
        
        remaining_display = str(q.total_remaining)
        if q.is_exhausted:
            remaining_display = '<span class="exhausted-text">EXHAUSTED</span>'
        
        error_display = ""
        if q.last_error:
            error_display = f'<br><span class="error-text">{q.last_error}</span>'
        
        rows.append(f"""
        <tr{exhausted_class}>
            <td class="owner">{q.owner} {priority_badge}</td>
            <td>{q.subscription}</td>
            <td>{free_bar}</td>
            <td>{overage_bar}</td>
            <td class="remaining">{remaining_display}</td>
            <td>{_format_reset_time(q.next_reset_epoch)}</td>
        </tr>{error_display}""")
    
    rows_html = "\n".join(rows) if rows else '<tr><td colspan="6" class="muted">No quota data yet. Waiting for first poll...</td></tr>'
    
    # Summary stats
    total = len(all_accounts)
    initialized = len([a for a in all_accounts if a.auth_manager is not None])
    polled = len(quota_list)
    exhausted_count = len([q for q in quota_list if q.is_exhausted])
    available_count = polled - exhausted_count
    
    priority_display = ", ".join(ACCOUNT_PRIORITY) if ACCOUNT_PRIORITY else "<span class='muted'>none (balanced mode)</span>"
    
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="refresh" content="30">
    <title>Kiro Gateway - Status Dashboard</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f1117; color: #e0e0e0; padding: 20px;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #fff; margin-bottom: 5px; font-size: 24px; }}
        .subtitle {{ color: #888; margin-bottom: 20px; font-size: 14px; }}
        .stats-grid {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 12px; margin-bottom: 24px;
        }}
        .stat-card {{
            background: #1a1d27; border-radius: 8px; padding: 16px;
            border: 1px solid #2a2d3a;
        }}
        .stat-label {{ color: #888; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }}
        .stat-value {{ color: #fff; font-size: 28px; font-weight: 600; margin-top: 4px; }}
        .stat-value.green {{ color: #4ade80; }}
        .stat-value.red {{ color: #f87171; }}
        .stat-value.orange {{ color: #fbbf24; }}
        table {{
            width: 100%; border-collapse: collapse;
            background: #1a1d27; border-radius: 8px; overflow: hidden;
            border: 1px solid #2a2d3a;
        }}
        th {{
            background: #222531; color: #a0a0a0; text-align: left;
            padding: 12px 16px; font-size: 12px; text-transform: uppercase;
            letter-spacing: 0.5px; border-bottom: 1px solid #2a2d3a;
        }}
        td {{ padding: 12px 16px; border-bottom: 1px solid #2a2d3a; font-size: 14px; }}
        tr:last-child td {{ border-bottom: none; }}
        tr.exhausted {{ opacity: 0.5; }}
        .owner {{ font-weight: 600; color: #fff; }}
        .remaining {{ font-weight: 600; text-align: center; }}
        .exhausted-text {{ color: #f87171; font-weight: 700; }}
        .error-text {{ color: #f87171; font-size: 12px; }}
        .badge {{
            display: inline-block; padding: 2px 8px; border-radius: 4px;
            font-size: 11px; font-weight: 600; text-transform: uppercase;
        }}
        .badge.priority {{ background: #3b82f6; color: #fff; }}
        .bar-container {{
            position: relative; width: 140px; height: 22px;
            background: #2a2d3a; border-radius: 4px; overflow: hidden;
        }}
        .bar-fill {{ height: 100%; border-radius: 4px; transition: width 0.3s; }}
        .bar-fill.green {{ background: #4ade80; }}
        .bar-fill.orange {{ background: #fbbf24; }}
        .bar-fill.red {{ background: #f87171; }}
        .bar-label {{
            position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
            font-size: 11px; color: #fff; text-shadow: 0 0 3px rgba(0,0,0,0.8);
            white-space: nowrap;
        }}
        .muted {{ color: #666; }}
        .footer {{ margin-top: 20px; text-align: center; color: #555; font-size: 12px; }}
        .footer a {{ color: #3b82f6; text-decoration: none; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Kiro Gateway Status</h1>
        <p class="subtitle">Uptime: {uptime} | Version: {APP_VERSION} | Auto-refresh: 30s</p>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Total Accounts</div>
                <div class="stat-value">{total}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Initialized</div>
                <div class="stat-value">{initialized}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Polled</div>
                <div class="stat-value">{polled}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Available</div>
                <div class="stat-value green">{available_count}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Exhausted</div>
                <div class="stat-value {'red' if exhausted_count > 0 else 'green'}">{exhausted_count}</div>
            </div>
        </div>
        
        <table>
            <thead>
                <tr>
                    <th>Owner</th>
                    <th>Subscription</th>
                    <th>Free Usage</th>
                    <th>Overage Usage</th>
                    <th>Total Remaining</th>
                    <th>Reset Date</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
        
        <p class="subtitle" style="margin-top: 16px;">
            Priority order: {priority_display} | Poll interval: {QUOTA_POLL_INTERVAL}s
        </p>
        
        <div class="footer">
            <a href="/api/quota">JSON API: /api/quota</a> |
            <a href="/api/status">JSON API: /api/status</a> |
            <a href="/health">Health: /health</a>
        </div>
    </div>
</body>
</html>"""
