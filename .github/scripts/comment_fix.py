import json
import os
import subprocess
from pathlib import Path

event_path = os.environ.get("GITHUB_EVENT_PATH", "")
pr_num = ""

if event_path and os.path.exists(event_path):
    with Path(event_path).open() as f:
        event = json.load(f)
        if "pull_request" in event:
            pr_num = str(event["pull_request"]["number"])

if not pr_num:
    result = subprocess.run(
        ["/usr/bin/gh", "pr", "view", os.environ.get("GITHUB_SHA", ""), "--json", "number", "-q", ".number"],
        capture_output=True,
        text=True,
        check=False,
    )
    pr_num = result.stdout.strip()

body = """## Fix Applied

Fixed:
- Added missing permissions to workflow jobs
- Applied pre-commit auto-fixes

Changes have been pushed to this PR.
"""

subprocess.run(["/usr/bin/gh", "pr", "comment", pr_num, "--body", body], check=True)
