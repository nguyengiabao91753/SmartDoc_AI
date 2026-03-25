import logging
import sys

LOG = logging.getLogger("smartdoc")
LOG.setLevel(logging.INFO)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
except Exception:
    pass

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
handler.setFormatter(formatter)
if not LOG.handlers:
    LOG.addHandler(handler)
