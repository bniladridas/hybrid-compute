import os
import subprocess

issue_url = os.environ.get("GITHUB_EVENT_PATH", "")
pr_num = issue_url.split("/")[-2] if issue_url else ""

body = """## Fix Applied

Fixed:
- Added missing permissions to workflow jobs
- Applied pre-commit auto-fixes

Changes have been pushed to this PR.
"""

subprocess.run(["gh", "pr", "comment", pr_num, "--body", body], check=True)
