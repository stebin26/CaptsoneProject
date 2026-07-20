"""Every @callback in the app lives under this package.

Dash's @callback decorator registers into a global registry, not onto the app
object, so callbacks work regardless of which module they sit in. That is what
lets layouts be rewritten freely without touching a line of logic -- provided
the component IDs in app/ids.py stay intact.

main.py imports this package once, at the very bottom of the file. Importing it
is what registers the callbacks; nothing here is called directly.
"""

from app.callbacks import (  # noqa: F401
    analytics,
    auth,
    business_intelligence,
    copilot,
    datasets,
    documents,
    evaluation,
    executive,
    feature_review,
    mapping_confirm,
    predictions,
    shell,
    upload,
)
