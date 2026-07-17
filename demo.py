#!/usr/bin/env python3
"""End-to-end demo / smoke test for the Industrial Operations Intelligence Platform.

Runs against a live stack (docker compose up). It logs in, proves RBAC by hitting
a protected endpoint, then walks every layer -- datasets, analytics, ML,
intelligence, executive summary, RAG, and (if Ollama is up) the AI copilot --
printing evidence at each step so a reviewer can watch the whole platform work
from one command.

    python scripts/demo.py
    python scripts/demo.py --email admin@ops.com --password ChangeMe123
    python scripts/demo.py --skip-agent      # skip the slow local-LLM step
    python scripts/demo.py --api http://localhost:8000
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

import requests

# ------------------------------------------------------------
# Tiny console helpers -- no dependencies beyond requests.
# ------------------------------------------------------------

_C = {
    "head": "\033[1;36m",   # bold cyan
    "ok": "\033[1;32m",     # bold green
    "warn": "\033[1;33m",   # bold yellow
    "err": "\033[1;31m",    # bold red
    "dim": "\033[2m",
    "reset": "\033[0m",
}


def _c(tag: str, text: str) -> str:
    return f"{_C[tag]}{text}{_C['reset']}"


def step(n: int, title: str) -> None:
    print(f"\n{_c('head', f'[{n}] {title}')}")
    print(_c("dim", "-" * 60))


def ok(msg: str) -> None:
    print(f"  {_c('ok', 'OK')}  {msg}")


def warn(msg: str) -> None:
    print(f"  {_c('warn', '!!')}  {msg}")


def fail(msg: str) -> None:
    print(f"  {_c('err', 'XX')}  {msg}")


def die(msg: str) -> None:
    fail(msg)
    print(_c("err", "\nDemo aborted."))
    sys.exit(1)


# ------------------------------------------------------------
# HTTP client -- bearer token carried across calls.
# ------------------------------------------------------------

class Client:
    def __init__(self, api_base: str) -> None:
        self.api = api_base.rstrip("/")
        self.v1 = f"{self.api}/api/v1"
        self.token: str | None = None
        self.refresh: str | None = None

    def _headers(self, auth: bool) -> dict[str, str]:
        if auth and self.token:
            return {"Authorization": f"Bearer {self.token}"}
        return {}

    def get(self, path: str, auth: bool = True, **kw: Any) -> requests.Response:
        return requests.get(f"{self.v1}{path}", headers=self._headers(auth),
                            timeout=kw.pop("timeout", (5, 240)), **kw)

    def post(self, path: str, auth: bool = True, **kw: Any) -> requests.Response:
        return requests.post(f"{self.v1}{path}", headers=self._headers(auth),
                             timeout=kw.pop("timeout", (5, 240)), **kw)


# ------------------------------------------------------------
# Steps
# ------------------------------------------------------------

def wait_for_health(cli: Client, retries: int = 30) -> None:
    step(1, "Preflight — is the API up?")
    url = f"{cli.api}/health"
    for i in range(retries):
        try:
            r = requests.get(url, timeout=(3, 5))
            if r.ok:
                ok(f"API healthy at {url}  ->  {r.json()}")
                return
        except requests.RequestException:
            pass
        print(_c("dim", f"  ...waiting for API ({i + 1}/{retries})"))
        time.sleep(3)
    die(f"API never became healthy at {url}. Is the stack running? (docker compose up -d)")


def prove_rbac_blocks_anon(cli: Client) -> None:
    step(2, "Security — a protected endpoint rejects anonymous callers")
    r = cli.get("/domains", auth=False)
    if r.status_code == 401:
        ok(f"GET /domains without a token -> 401 {r.json().get('detail')!r}")
    else:
        warn(f"Expected 401, got {r.status_code}. RBAC may not be enforced here.")


def do_login(cli: Client, email: str, password: str) -> dict[str, Any]:
    step(3, "Auth — log in and load identity")
    r = cli.post("/auth/login", auth=False, json={"email": email, "password": password})
    if not r.ok:
        die(f"Login failed ({r.status_code}): {r.text}")
    tokens = r.json()
    cli.token = tokens["access_token"]
    cli.refresh = tokens["refresh_token"]
    ok("Logged in — received access + refresh tokens")

    me = cli.get("/auth/me")
    if not me.ok:
        die(f"/auth/me failed ({me.status_code}): {me.text}")
    who = me.json()
    ok(f"Identity: {who['email']}  roles={who['roles']}")
    ok(f"Permissions ({len(who['permissions'])}): {', '.join(who['permissions'])}")
    return who


def prove_rbac_allows_authed(cli: Client) -> None:
    step(4, "Security — the same endpoint now succeeds with a token")
    r = cli.get("/domains")
    if r.ok:
        ok(f"GET /domains with a token -> 200, {len(r.json())} domains")
    else:
        warn(f"Authenticated call failed ({r.status_code}): {r.text}")


def pick_dataset(cli: Client) -> int | None:
    step(5, "Data hub — list onboarded datasets")
    r = cli.get("/datasets")
    if not r.ok:
        die(f"/datasets failed ({r.status_code}): {r.text}")
    datasets = r.json()
    if not datasets:
        warn("No datasets found. The seed step may not have run.")
        return None
    for d in datasets:
        print(_c("dim",
                 f"     #{d['dataset_id']:<3} {d['business_name']:<22} "
                 f"{d.get('industry') or '-':<14} "
                 f"collected={d['features_collected']} skipped={d['features_skipped']}"))
    chosen = datasets[0]["dataset_id"]
    ok(f"Using dataset #{chosen} ({datasets[0]['business_name']}) for the walkthrough")
    return chosen


def show_analytics(cli: Client, ds: int) -> None:
    step(6, "Analytics — Spark-computed KPIs and trends")
    r = cli.get(f"/analytics/{ds}/metrics")
    if r.ok:
        metrics = r.json()
        ok(f"{len(metrics)} metric summaries across domains")
        for m in metrics[:5]:
            print(_c("dim",
                     f"     {m.get('domain','?'):<12} {m.get('metric_name','?'):<20} "
                     f"avg={m.get('metric_avg')}"))
    else:
        warn(f"analytics metrics ({r.status_code}) — has the analytics DAG run for this dataset?")


def show_ml(cli: Client, ds: int) -> None:
    step(7, "ML — forecasts, anomalies, risk")
    fr = cli.get(f"/ml/{ds}/forecasts")
    an = cli.get(f"/ml/{ds}/anomalies", params={"limit": 500})
    rk = cli.get(f"/ml/{ds}/risk-scores")
    if fr.ok:
        ok(f"{len(fr.json())} forecast points")
    else:
        warn(f"forecasts ({fr.status_code})")
    if an.ok:
        anomalies = an.json()
        highs = sum(1 for a in anomalies if a.get("severity") == "high")
        ok(f"{len(anomalies)} anomalies ({highs} high severity)")
    else:
        warn(f"anomalies ({an.status_code})")
    if rk.ok:
        ok(f"{len(rk.json())} entity risk scores")
    else:
        warn(f"risk-scores ({rk.status_code})")


def show_intelligence(cli: Client, ds: int) -> None:
    step(8, "Intelligence — cross-domain relationships")
    r = cli.get(f"/intelligence/{ds}", timeout=(5, 180))
    if r.ok:
        data = r.json()
        insights = data.get("insights") or data.get("relationships") or []
        ok(f"Intelligence engine returned {len(insights)} cross-domain signal(s)")
    else:
        warn(f"intelligence ({r.status_code})")


def show_executive(cli: Client, ds: int) -> None:
    step(9, "Executive summary — the whole operation in one call")
    r = cli.get(f"/executive/{ds}/summary")
    if not r.ok:
        warn(f"executive summary ({r.status_code})")
        return
    s = r.json()
    idx = s.get("risk_index", {})
    ok(f"Business: {s.get('business_name')}  ({s.get('industry') or 'n/a'})")
    ok(f"Operational Risk Index: {idx.get('value')} ({idx.get('band')}) — "
       f"{s.get('active_domain_count')} active domains")
    ok(f"Open alerts: {s.get('open_alert_count')}  |  entities at risk: {s.get('entities_at_risk')}"
       f"  |  insights: {s.get('insight_count')}")


def show_rag(cli: Client, ds: int) -> None:
    step(10, "RAG — document assistant (grounded Q&A)")
    r = cli.get(f"/rag/{ds}/documents")
    if not r.ok:
        warn(f"rag documents ({r.status_code})")
        return
    docs = r.json()
    ok(f"{len(docs)} document(s) indexed for this dataset")
    if not docs:
        print(_c("dim", "     (upload a manual on the Documents page to see grounded answers)"))


def ask_agent(cli: Client, ds: int) -> None:
    step(11, "AI Copilot — natural-language investigation (local LLM)")
    h = cli.get("/agent/health", timeout=(3, 10))
    if not h.ok or not h.json().get("llm_reachable"):
        warn("Agent LLM not reachable (Ollama not running?). Skipping the copilot step.")
        print(_c("dim", "     Start Ollama with llama3.2:3b to enable this, or use --skip-agent."))
        return
    ok(f"Agent ready — model {h.json().get('model')}, {h.json().get('tool_count')} tools")

    question = "Which domain has the highest risk, and what is driving it?"
    print(_c("dim", f"     Asking: {question!r}"))
    print(_c("dim", "     (a multi-step loop on a 3B CPU model — this can take a minute)"))
    r = cli.post("/agent/ask", json={"question": question, "dataset_id": ds},
                 timeout=(5, 240))
    if not r.ok:
        warn(f"agent ask ({r.status_code}): {r.text[:200]}")
        return
    a = r.json()
    ok(f"Answered in {a.get('elapsed_seconds')}s using {a.get('steps')} step(s)")
    ok(f"Tools consulted: {', '.join(a.get('tools_used') or []) or 'none'}")
    print(_c("dim", "\n     Answer:"))
    print("     " + (a.get("answer") or "").replace("\n", "\n     "))


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="End-to-end demo of the platform.")
    p.add_argument("--api", default="http://localhost:8000", help="API base URL")
    p.add_argument("--email", default="admin@ops.com")
    p.add_argument("--password", default="ChangeMe123")
    p.add_argument("--dataset", type=int, default=None,
                   help="Force a dataset id (default: first available)")
    p.add_argument("--skip-agent", action="store_true",
                   help="Skip the slow local-LLM copilot step")
    args = p.parse_args()

    print(_c("head", "\n=========================================================="))
    print(_c("head", " Industrial Operations Intelligence Platform — Live Demo"))
    print(_c("head", "=========================================================="))

    cli = Client(args.api)

    wait_for_health(cli)
    prove_rbac_blocks_anon(cli)
    do_login(cli, args.email, args.password)
    prove_rbac_allows_authed(cli)

    ds = args.dataset or pick_dataset(cli)
    if ds is None:
        die("No dataset to demo. Re-run the stack so the seed step populates data.")

    show_analytics(cli, ds)
    show_ml(cli, ds)
    show_intelligence(cli, ds)
    show_executive(cli, ds)
    show_rag(cli, ds)
    if not args.skip_agent:
        ask_agent(cli, ds)

    print(_c("ok", "\n==========================================================")); 
    print(_c("ok", " Demo complete — every layer exercised end to end."))
    print(_c("ok", "=========================================================="))


if __name__ == "__main__":
    main()