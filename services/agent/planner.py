"""Planner placeholder -- intentionally unused.

An explicit LLM planning step was trialled and dropped: the local 3B model could
not reliably follow a multi-step plan, and every failure compounded across the
steps that followed. The ReAct loop with memory and tool pre-selection proved
both more accurate and faster. This module is kept so the decision stays visible
in the codebase rather than being silently forgotten.
"""
