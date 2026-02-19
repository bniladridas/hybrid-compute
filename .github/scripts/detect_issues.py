import os

import yaml

SKIP = ["needs", "if", "runs-on", "environment", "timeout-minutes", "continue-on-error"]
ISSUES = ""

for root, dirs, files in os.walk(".github/workflows"):
    for fname in files:
        if fname.endswith(".yml"):
            fpath = os.path.join(root, fname)
            with open(fpath) as f:
                workflow = yaml.safe_load(f)

            if "jobs" in workflow:
                for job_name, job_config in workflow["jobs"].items():
                    if job_name in SKIP:
                        continue
                    if "permissions" in job_config:
                        continue
                    if ISSUES:
                        ISSUES += "\n"
                    ISSUES += f"Job '{job_name}' in '{fname}' missing permissions"

with open(os.environ["GITHUB_OUTPUT"], "a") as f:
    f.write(f"issues_found={'true' if ISSUES else 'false'}\n")
    if ISSUES:
        f.write(f"ISSUES<<EOF\n{ISSUES}\nEOF\n")

if ISSUES:
    with open("/tmp/issues.txt", "w") as f:
        f.write(ISSUES)
