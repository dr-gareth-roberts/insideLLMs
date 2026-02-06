"""Allow ``python -m insideLLMs.cli`` to work."""

import sys

from insideLLMs.cli import main

sys.exit(main())
