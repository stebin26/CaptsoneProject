"""Single source of truth for every Dash component ID in the dashboard.

The app runs with suppress_callback_exceptions=True. If a layout stops
rendering a component that a callback references, Dash does not raise --
the callback simply never fires. The page renders, looks fine, and lies.

Import from here instead of typing string literals, so the ID-assertion
test can prove that layouts and callbacks still agree after a rewrite.

Flat IDs are constants. Pattern-matching (dict) IDs are builder functions --
call them, never inline the dict, so key names cannot drift.
"""

from __future__ import annotations

from typing import Any

# ============================================================
# Global -- defined in main.py, referenced across pages
# ============================================================

URL = "url"
ONBOARDING_STORE = "onboarding-store"
REVIEW_REFRESH = "review-refresh"


# ============================================================
# App shell
# ============================================================

SHELL_ROOT = "shell-root"
SHELL_SIDEBAR = "shell-sidebar"
SHELL_NAV = "shell-nav"
SHELL_SIDEBAR_TOGGLE = "shell-sidebar-toggle"
SHELL_COLLAPSED = "shell-collapsed"          # dcc.Store, storage_type="local"
SHELL_SCRIM = "shell-scrim"
SHELL_MAIN = "shell-main"
SHELL_INIT = "shell-init"
SHELL_PAGE_WRAP = "shell-page-wrap"          # wraps page_container; hidden on 403
SHELL_GUARD_403 = "shell-guard-403"          # permission-denied overlay

TOPBAR_TITLE = "topbar-title"
TOPBAR_SUBTITLE = "topbar-subtitle"
TOPBAR_DATASET = "topbar-dataset"
TOPBAR_DATE_RANGE = "topbar-date-range"
TOPBAR_REFRESH = "topbar-refresh"
TOPBAR_LAST_REFRESH = "topbar-last-refresh"
TOPBAR_BELL = "topbar-bell"
TOPBAR_BELL_COUNT = "topbar-bell-count"
TOPBAR_BELL_PANEL = "topbar-bell-panel"
TOPBAR_PROFILE = "topbar-profile"
TOPBAR_LOGOUT = "topbar-logout"
TOPBAR_USER_NAME = "topbar-user-name"

# Global state, owned by the shell and read by every page.
ACTIVE_DATASET = "active-dataset"            # dcc.Store, session
ACTIVE_DATE_RANGE = "active-date-range"      # dcc.Store, session
REFRESH_TOKEN = "refresh-token"              # dcc.Store, memory


# Auth state (Item 6), owned by the shell, read by the auth guard + nav.
ACCESS_TOKEN = "access-token"                # dcc.Store, session (JWT access)
AUTH_REFRESH_TOKEN = "auth-refresh-token"    # dcc.Store, session (opaque refresh)
AUTH_USER = "auth-user"                      # dcc.Store, session (id/email/roles/perms)
AUTH_GUARD = "auth-guard"                    # dummy output for the redirect callback

# ============================================================
# Executive  --  /
# ============================================================

EXEC_INIT = "exec-init"
EXEC_STORE = "exec-store"
EXEC_KPI_INDUSTRY = "exec-kpi-industry"
EXEC_KPI_RISK = "exec-kpi-risk"
EXEC_KPI_ALERTS = "exec-kpi-alerts"
EXEC_KPI_FRESHNESS = "exec-kpi-freshness"
EXEC_DOMAIN_HEALTH = "exec-domain-health"
EXEC_TOP_RISKS = "exec-top-risks"
EXEC_ACTIVE_ALERTS = "exec-active-alerts"
EXEC_FORECASTS = "exec-forecasts"
EXEC_INSIGHTS = "exec-insights"
EXEC_ERROR = "exec-error"

# ============================================================
# Login  --  /login
# ============================================================

LOGIN_EMAIL = "login-email"
LOGIN_PASSWORD = "login-password"
LOGIN_SUBMIT = "login-submit"
LOGIN_STATUS = "login-status"
LOGIN_GOOGLE = "login-google"

# ============================================================
# Upload  --  /upload
# ============================================================

UPLOAD_BUSINESS_NAME = "upload-business-name"
UPLOAD_INDUSTRY = "upload-industry"
UPLOAD_DATA = "upload-data"
UPLOAD_FILENAME = "upload-filename"
UPLOAD_SUBMIT = "upload-submit"
UPLOAD_STATUS = "upload-status"


# ============================================================
# Mapping confirm  --  /confirm
# ============================================================

CONFIRM_HEADER = "confirm-header"
CONFIRM_COLUMNS = "confirm-columns"
CONFIRM_SUBMIT = "confirm-submit"
CONFIRM_STATUS = "confirm-status"


def confirm_role(column: str) -> dict[str, Any]:
    return {"type": "confirm-role", "column": column}


def confirm_domain(column: str) -> dict[str, Any]:
    return {"type": "confirm-domain", "column": column}


def confirm_metric(column: str) -> dict[str, Any]:
    return {"type": "confirm-metric", "column": column}


# ============================================================
# Feature review  --  /review
# ============================================================

REVIEW_INIT = "review-init"
REVIEW_HEADER = "review-header"
REVIEW_COLLECTED = "review-collected"
REVIEW_MISSED = "review-missed"
REVIEW_COVERAGE_CHART = "review-coverage-chart"
REVIEW_DATA_CHARTS = "review-data-charts"
REVIEW_ADD_STATUS = "review-add-status"


def add_button(column: str) -> dict[str, Any]:
    return {"type": "add-button", "column": column}


def add_domain(column: str) -> dict[str, Any]:
    return {"type": "add-domain", "column": column}


def add_metric(column: str) -> dict[str, Any]:
    return {"type": "add-metric", "column": column}


# ============================================================
# Datasets  --  /datasets
# ============================================================

DATASETS_INIT = "datasets-init"
DATASETS_LIST = "datasets-list"


def open_dataset(dataset_id: int) -> dict[str, Any]:
    return {"type": "open-dataset", "id": dataset_id}


# ============================================================
# Analytics  --  /analytics
# ============================================================

ANALYTICS_INIT = "analytics-init"
ANALYTICS_DATASET = "analytics-dataset"
ANALYTICS_METRIC = "analytics-metric"
ANALYTICS_SUMMARY = "analytics-summary"
ANALYTICS_METRICS_CHART = "analytics-metrics-chart"
ANALYTICS_TREND_CHART = "analytics-trend-chart"
ANALYTICS_FEATURES_TABLE = "analytics-features-table"
ANALYTICS_DOMAIN_STATUS = "analytics-domain-status"
ANALYTICS_DOMAIN_CHARTS = "analytics-domain-charts"


# ============================================================
# Predictions  --  /predictions
# ============================================================

PRED_INIT = "pred-init"
PRED_DATASET = "pred-dataset"
PRED_KPIS = "pred-kpis"
PRED_TABLE = "pred-table"
PRED_DOMAIN_CHARTS = "pred-domain-charts"


# ============================================================
# Intelligence  --  /intelligence   (file: business_intelligence.py)
# ============================================================

BI_INIT = "bi-init"
BI_DATASET = "bi-dataset"
BI_SUMMARY = "bi-summary"
BI_INSIGHTS = "bi-insights"


# ============================================================
# Documents  --  /documents
# ============================================================

DOC_INIT = "doc-init"
DOC_DATASET = "doc-dataset"
DOC_UPLOAD = "doc-upload"
DOC_UPLOAD_STATUS = "doc-upload-status"
DOC_LIST = "doc-list"
DOC_POLL = "doc-poll"
DOC_QUESTION = "doc-question"
DOC_ASK = "doc-ask"
DOC_ANSWER = "doc-answer"


# ============================================================
# Copilot  --  /copilot
# ============================================================

COPILOT_INIT = "copilot-init"
COPILOT_DATASET = "copilot-dataset"
COPILOT_SESSION = "copilot-session"
COPILOT_HISTORY = "copilot-history"
COPILOT_TRANSCRIPT = "copilot-transcript"
COPILOT_INPUT = "copilot-input"
COPILOT_SEND = "copilot-send"
COPILOT_PENDING = "copilot-pending"
COPILOT_STATUS = "copilot-status"
COPILOT_SUGGESTIONS = "copilot-suggestions"   # static container, no callback


def copilot_suggestion(index: int) -> dict[str, Any]:
    return {"type": "copilot-suggestion", "index": index}


# ============================================================
# Pattern-matching type names
# ============================================================
# The ID test walks the component tree and cannot call the builders above,
# so it needs the raw "type" strings. This is the one unavoidable duplication.

PATTERN_TYPES: frozenset[str] = frozenset(
    {
        "confirm-role",
        "confirm-domain",
        "confirm-metric",
        "add-button",
        "add-domain",
        "add-metric",
        "open-dataset",
        "copilot-suggestion",
    }
)


# ============================================================
# IDs that intentionally have no callback
# ============================================================
# Everything else must be referenced by at least one callback, or the
# ID-assertion check fails. Add here only with a reason.

CALLBACK_FREE: frozenset[str] = frozenset(
    {
        COPILOT_SUGGESTIONS,   # static container rendered inside the layout
        SHELL_ROOT,            # collapse toggles a class on it clientside via
                               # getElementById -- no Dash callback references it
        SHELL_MAIN,            # layout anchor for page_container            # layout anchor for page_container
        SHELL_SIDEBAR,         # the <aside> wrapper; collapse toggles a class
                               # on SHELL_ROOT and the nav rebuilds via SHELL_NAV
        TOPBAR_PROFILE,        # static until auth lands (Item 6)
    }
)