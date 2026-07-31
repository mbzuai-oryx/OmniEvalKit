#!/usr/bin/env python3

import sys
from pathlib import Path


PACKAGE_PARENT = Path(__file__).resolve().parent.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from End_to_End_Model.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
