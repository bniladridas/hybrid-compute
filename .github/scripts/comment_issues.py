import os
import subprocess

pr_num = (
    os.environ.get("GITHUB_REF", "").split("/")[-1]
    if "refs/pull" in os.environ.get("GITHUB_REF", "")
    else os.environ.get("GITHUB_EVENT_NUMBER", "")
)

try:
    with open("/tmp/issues.txt") as f:
        issues = f.read()
except FileNotFoundError:
    issues = os.environ.get("ISSUES", "")

if not pr_num:
    pr_num = subprocess.run(
        ["gh", "pr", "view", os.environ.get("GITHUB_SHA", ""), "--json", "number", "-q", ".number"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()

body = f"""## PR Analysis

I found issues in this PR:

{issues}

### Fix Commands

- /fix-workflow-permissions - Fix permissions
- /fix precommit - Fix pre-commit issues
- /fix all - Apply all fixes
"""

subprocess.run(["gh", "pr", "comment", pr_num, "--body", body], check=True)
