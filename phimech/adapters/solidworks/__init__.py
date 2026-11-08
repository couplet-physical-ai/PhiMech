"""Validated neutral-file handoff; no COM dependency or model execution."""

from .bundle import prepare_bundle, validate_bundle

__all__ = ["prepare_bundle", "validate_bundle"]


def doctor_section() -> dict:
    """Report the portable capability without advertising a native installation."""
    return {"ok": True, "native_available": False, "native_tested": False,
            "model_execution": "disabled",
            "note": "STEP handoff contract available; native Windows capture untested"}
