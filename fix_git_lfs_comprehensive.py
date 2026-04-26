#!/usr/bin/env python3
"""
Comprehensive Git LFS fix script.
Removes large files from git history and sets up proper LFS tracking.
"""

import subprocess
import os
import sys
from pathlib import Path

def run_command(cmd, shell=False):
    """Run a shell command and return the result."""
    print(f"▶ Running: {cmd}")
    try:
        result = subprocess.run(
            cmd,
            shell=shell,
            capture_output=True,
            text=True,
            check=False
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr and result.returncode != 0:
            print(f"⚠ {result.stderr}", file=sys.stderr)
        return result.returncode
    except Exception as e:
        print(f"✗ Error: {e}")
        return 1

def main():
    repo_path = Path(__file__).parent
    os.chdir(repo_path)
    
    print("=" * 80)
    print("COMPREHENSIVE Git LFS Fix - Remove files from history and re-track with LFS")
    print("=" * 80)
    
    # Large files to handle
    large_files = [
        "notebooks/01_loan_default_EDA.ipynb",
        "data/raw/accepted_2007_to_2018Q4.csv",
        "data/raw/rejected_2007_to_2018Q4.csv"
    ]
    
    # Step 1: Check if git-filter-repo is installed
    print("\n[1/6] Checking for git-filter-repo...")
    if run_command("git filter-repo --version") != 0:
        print("⚠ git-filter-repo not found. Installing...")
        run_command("pip install git-filter-repo", shell=True)
    
    # Step 2: Create backup
    print("\n[2/6] Creating backup of current state...")
    run_command("git log --oneline -5")
    
    # Step 3: Remove large files from history
    print("\n[3/6] Removing large files from git history...")
    for file in large_files:
        print(f"\n  Removing {file} from history...")
        run_command(f'git filter-repo --path "{file}" --invert-paths --force', shell=True)
    
    # Step 4: Verify files still exist on disk
    print("\n[4/6] Verifying files still exist on disk...")
    for file in large_files:
        file_path = repo_path / file
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ {file} ({size_mb:.2f} MB)")
        else:
            print(f"  ✗ {file} NOT FOUND")
    
    # Step 5: Set up .gitattributes
    print("\n[5/6] Setting up .gitattributes for LFS...")
    gitattributes_path = repo_path / ".gitattributes"
    
    lfs_patterns = [
        "*.ipynb filter=lfs diff=lfs merge=lfs -text",
        "*.csv filter=lfs diff=lfs merge=lfs -text"
    ]
    
    with open(gitattributes_path, "w") as f:
        f.write("# Git LFS configuration\n")
        for pattern in lfs_patterns:
            f.write(f"{pattern}\n")
    
    print("✓ .gitattributes updated")
    
    # Step 6: Re-add files with LFS tracking
    print("\n[6/6] Re-adding files with LFS tracking...")
    
    # Initialize LFS
    run_command("git lfs install --local")
    
    # Add files with LFS
    for file in large_files:
        file_path = repo_path / file
        if file_path.exists():
            print(f"  Tracking {file} with LFS...")
            run_command(f'git lfs track "{file}"', shell=True)
            run_command(f'git add "{file}"', shell=True)
    
    # Add .gitattributes
    run_command("git add .gitattributes", shell=True)
    
    # Commit
    print("\nCreating commit with LFS-tracked files...")
    run_command('git commit -m "Re-add large files tracked with Git LFS"', shell=True)
    
    print("\n" + "=" * 80)
    print("✓ Git LFS setup complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Review changes: git log --oneline -3")
    print("2. Push to GitHub: git push origin main --force")
    print("\nWARNING: Using --force because we rewrote history with git filter-repo")
    print("=" * 80)

if __name__ == "__main__":
    sys.exit(main())
