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
    github_ref = os.environ.get("GITHUB_REF", "")
    if "pull" in github_ref:
        parts = github_ref.split("/")
        for i, part in enumerate(parts):
            if part == "pull" and i + 1 < len(parts):
                pr_num = parts[i + 1]
                break

if not pr_num:
    result = subprocess.run(
        ["/usr/bin/gh", "pr", "view", os.environ.get("GITHUB_SHA", ""), "--json", "number", "-q", ".number"],
        capture_output=True,
        text=True,
        check=False,
    )
    pr_num = result.stdout.strip()

issues = ""
issues_path = Path("/tmp/issues.txt")
issues = issues_path.read_text() if issues_path.exists() else os.environ.get("ISSUES", "")

body = f"""## PR Analysis

I found issues in this PR:

{issues}

### Fix Commands

- /fix-workflow-permissions - Fix permissions
- /fix precommit - Fix pre-commit issues
- /fix all - Apply all fixes
"""

subprocess.run(["/usr/bin/gh", "pr", "comment", pr_num, "--body", body], check=True)
