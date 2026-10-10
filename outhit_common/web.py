import json
import math
from typing import Any

import numpy as np
from fastapi.responses import JSONResponse


def clean(value: Any) -> Any:
    """Make analytics output JSON-safe: numpy scalars -> Python, NaN/inf -> null."""
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class CleanJSONResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        return json.dumps(clean(content), ensure_ascii=False, separators=(",", ":"), default=str).encode()
