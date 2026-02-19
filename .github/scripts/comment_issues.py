import os
import json
import subprocess

# Get PR number from event file
event_path = os.environ.get("GITHUB_EVENT_PATH", "")
pr_num = ""

if event_path and os.path.exists(event_path):
    with open(event_path) as f:
        event = json.load(f)
        if "pull_request" in event:
            pr_num = str(event["pull_request"]["number"])

# Fallback: try from GITHUB_REF
if not pr_num:
    github_ref = os.environ.get("GITHUB_REF", "")
    if "pull" in github_ref:
        parts = github_ref.split("/")
        for i, part in enumerate(parts):
            if part == "pull" and i + 1 < len(parts):
                pr_num = parts[i + 1]
                break

# Fallback: try from gh cli
if not pr_num:
    result = subprocess.run(
        ["gh", "pr", "view", os.environ.get("GITHUB_SHA", ""), "--json", "number", "-q", ".number"],
        capture_output=True,
        text=True,
        check=False,
    )
    pr_num = result.stdout.strip()

try:
    with open("/tmp/issues.txt") as f:
        issues = f.read()
except FileNotFoundError:
    issues = os.environ.get("ISSUES", "")

body = f"""## PR Analysis

I found issues in this PR:

{issues}

### Fix Commands

- /fix-workflow-permissions - Fix permissions
- /fix precommit - Fix pre-commit issues
- /fix all - Apply all fixes
"""

subprocess.run(["gh", "pr", "comment", pr_num, "--body", body], check=True)
