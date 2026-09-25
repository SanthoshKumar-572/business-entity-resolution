"""
push_to_github.py
==================
Helper script to push code to GitHub.
Run this from the student_resource/ directory.

Usage:
  python3 code/push_to_github.py "Your commit message"

This will:
1. Initialize git if needed
2. Set remote origin
3. Add code files only (respecting .gitignore)
4. Commit and push

Prerequisites:
  - Git must be installed and in PATH
  - GitHub credentials configured (token or SSH)
"""

import os
import sys
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_URL = "https://github.com/SanthoshKumar-572/business-entity-resolution.git"


def run(cmd: list, cwd: str = ROOT, check: bool = True) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=False, text=True)
    if check and result.returncode != 0:
        print(f"  ERROR: command failed (code {result.returncode})")
        sys.exit(1)
    return result


def main():
    msg = sys.argv[1] if len(sys.argv) > 1 else "Update pipeline code"

    print("=" * 60)
    print("GitHub Push Helper")
    print("=" * 60)

    # Check git
    result = subprocess.run(["git", "--version"], capture_output=True, text=True)
    if result.returncode != 0:
        print("ERROR: git not found in PATH.")
        print("Install git from https://git-scm.com/download/win")
        sys.exit(1)
    print(f"  git version: {result.stdout.strip()}")

    # Init git if needed
    git_dir = os.path.join(ROOT, ".git")
    if not os.path.isdir(git_dir):
        print("\n[Init] Initializing git repository ...")
        run(["git", "init"])
        run(["git", "remote", "add", "origin", REPO_URL])
    else:
        # Ensure remote is set
        r = subprocess.run(["git", "remote", "get-url", "origin"], cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            run(["git", "remote", "add", "origin", REPO_URL])
        else:
            print(f"  Remote: {r.stdout.strip()}")

    # Config
    run(["git", "config", "user.email", "santhosh@team.local"])
    run(["git", "config", "user.name", "Person2-Laptop"])

    # Add files (only code, not data)
    print("\n[Add] Staging code files ...")
    run(["git", "add", "code/", "requirements.txt", "README.md", ".gitignore",
         "utils/validate_submission.py", "Documentation_template.md"])

    # Check status
    status = subprocess.run(["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True)
    if not status.stdout.strip():
        print("  Nothing to commit.")
        return 0

    print(f"  Changes:\n{status.stdout}")

    # Commit
    print(f"\n[Commit] '{msg}' ...")
    r = subprocess.run(
        ["git", "commit", "-m", msg],
        cwd=ROOT, capture_output=False, text=True
    )
    if r.returncode != 0:
        print("  Nothing new to commit.")
        return 0

    # Push
    print("\n[Push] Pushing to GitHub ...")
    run(["git", "push", "-u", "origin", "main"], check=False)
    # Try master if main fails
    r2 = subprocess.run(["git", "push", "-u", "origin", "master"], cwd=ROOT, capture_output=False, text=True)

    print("\n[Done] Push complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
