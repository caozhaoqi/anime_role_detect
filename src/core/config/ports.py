"""Port env-var coercion helper.

Kubernetes automatically injects ``<SERVICE_NAME>_PORT`` (and
``<SERVICE_NAME>_PORT_<PORT>_TCP``) environment variables for every Service
into Pods. Their value is a URI like ``tcp://10.96.0.1:8000`` — **not** a
plain integer. Code that reads these vars with ``int(os.environ.get(...))`` or
with a pydantic ``int`` field therefore crashes at startup inside the cluster.

This helper extracts the port integer from either a plain ``"8000"`` string or a
``"tcp://host:8000"`` / ``"tcp://[::1]:8000"`` URI, falling back to ``default``
when the value cannot be parsed.
"""

import re
from typing import Optional, Union

_PORT_RE = re.compile(r":(\d+)\s*$")


def coerce_port(value: Union[None, int, str], default: int) -> int:
    """Return an integer port from ``value``, tolerating K8s ``tcp://`` URIs.

    Args:
        value: Raw environment value (None, int, or str).
        default: Port to use when ``value`` is None or unparseable.

    Returns:
        Parsed integer port.
    """
    if value is None:
        return default
    if isinstance(value, bool):  # guard: bool is a subclass of int
        return default
    if isinstance(value, int):
        return value
    s = str(value).strip()
    if not s:
        return default
    # K8s injects tcp://10.96.0.1:8000 — pull the trailing :port
    m = _PORT_RE.search(s)
    if m:
        return int(m.group(1))
    # Plain integer string
    try:
        return int(s)
    except ValueError:
        return default
