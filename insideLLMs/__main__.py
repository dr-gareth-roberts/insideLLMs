"""Allow ``python -m insideLLMs`` to work (the standard module entry point).

The CLI implementation lives in :mod:`insideLLMs.cli`; this module only
delegates to it so both invocation styles behave identically:

- ``python -m insideLLMs``
- ``python -m insideLLMs.cli``
- the ``insidellms`` console script
"""

import sys

from insideLLMs.cli import main

sys.exit(main())
