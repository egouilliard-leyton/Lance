"""Allow ``python -m live_meeting`` to invoke the CLI."""

import sys

from live_meeting.cli import main

sys.exit(main())
