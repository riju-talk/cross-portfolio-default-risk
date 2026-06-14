#!/usr/bin/env python3
"""
nb2md.py — Bidirectional Jupyter Notebook ↔ Markdown Converter

Converts all .ipynb files in the CURRENT DIRECTORY to .md (and vice versa).
Does NOT recurse into subdirectories.

Usage:
    python nb2md.py --to-md          # Convert all .ipynb → .md
    python nb2md.py --to-nb          # Convert all .md → .ipynb
    python nb2md.py --to-md --force  # Overwrite existing .md files
    python nb2md.py --to-nb --force  # Overwrite existing .ipynb files
    python nb2md.py --to-md --dry-run # Preview without converting

Dependencies:
    pip install jupytext>=1.15.0

If jupytext is unavailable, falls back to basic nbformat + markdown conversion
(preserves code cells only, limited metadata).
"""

import argparse
import json
import os
import sys
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────
CURRENT_DIR = Path.cwd()
ENCODING = "utf-8"

# ── Jupytext backend (preferred) ───────────────────────────────────────────

def _has_jupytext() -> bool:
    try:
        import jupytext
        return True
    except ImportError:
        return False


def _convert_with_jupytext(src: Path, dst: Path, to_format: str) -> None:
    """Use jupytext for high-fidelity conversion."""
    import jupytext

    with open(src, "r", encoding=ENCODING) as f:
        notebook = jupytext.read(f)

    with open(dst, "w", encoding=ENCODING) as f:
        jupytext.write(notebook, f, fmt=to_format)


# ── Fallback backend (nbformat + manual markdown) ─────────────────────────

def _fallback_ipynb_to_md(src: Path, dst: Path) -> None:
    """Convert .ipynb to markdown without jupytext. Preserves code & outputs."""
    import nbformat

    with open(src, "r", encoding=ENCODING) as f:
        nb = nbformat.read(f, as_version=4)

    lines = []
    lines.append(f"# {nb.metadata.get('title', src.stem)}\n\n")

    for i, cell in enumerate(nb.cells):
        cell_type = cell.cell_type
        source = cell.source.strip()

        if not source and cell_type == "markdown":
            continue

        if cell_type == "markdown":
            lines.append(f"{source}\n\n")

        elif cell_type == "code":
            # Code fence with language hint
            lines.append(f"``` python\n{source}\n```\n\n")

            # Capture outputs
            outputs = cell.get("outputs", [])
            if outputs:
                out_lines = []
                for out in outputs:
                    ot = out.output_type
                    if ot == "stream":
                        out_lines.append(out.text)
                    elif ot in ("execute_result", "display_data"):
                        data = out.get("data", {})
                        if "text/plain" in data:
                            out_lines.append(str(data["text/plain"]))
                        if "image/png" in data:
                            out_lines.append("[PNG image output]")
                        if "text/html" in data:
                            out_lines.append("[HTML output]")
                    elif ot == "error":
                        ename = out.get("ename", "Error")
                        evalue = out.get("evalue", "")
                        out_lines.append(f"{ename}: {evalue}")

                if out_lines:
                    joined = "\n".join(out_lines).strip()
                    lines.append(f"**Output:**\n\n```\n{joined}\n```\n\n")

        elif cell_type == "raw":
            lines.append(f"```\n{source}\n```\n\n")

    # Embed notebook metadata as YAML frontmatter
    meta = {
        "kernelspec": nb.metadata.get("kernelspec", {}),
        "language_info": nb.metadata.get("language_info", {}),
    }
    frontmatter = f"---\n{json.dumps(meta, indent=2)}\n---\n\n"

    with open(dst, "w", encoding=ENCODING) as f:
        f.write(frontmatter + "".join(lines))


def _fallback_md_to_ipynb(src: Path, dst: Path) -> None:
    """Convert markdown back to .ipynb without jupytext. Best-effort."""
    import nbformat

    with open(src, "r", encoding=ENCODING) as f:
        content = f.read()

    # Parse YAML frontmatter if present
    metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
    if content.startswith("---"):
        try:
            end = content.find("---", 3)
            if end != -1:
                meta_json = content[3:end].strip()
                metadata.update(json.loads(meta_json))
                content = content[end + 3:].strip()
        except json.JSONDecodeError:
            pass

    nb = nbformat.v4.new_notebook(metadata=metadata)

    # Simple parsing: split by code fences
    import re
    pattern = r'```\s*(\w*)\n(.*?)\n```'
    parts = re.split(pattern, content, flags=re.DOTALL)

    # parts alternates: text, lang, code, text, lang, code, ...
    cells = []
    i = 0
    while i < len(parts):
        text = parts[i].strip()
        if text:
            cells.append(nbformat.v4.new_markdown_cell(text))
        i += 1
        if i < len(parts) - 1:
            lang = parts[i].strip()
            code = parts[i + 1].strip()
            if lang.lower() in ("python", "py", ""):
                cells.append(nbformat.v4.new_code_cell(code))
            else:
                cells.append(nbformat.v4.new_raw_cell(code))
            i += 2
        else:
            i += 1

    nb.cells = cells

    with open(dst, "w", encoding=ENCODING) as f:
        nbformat.write(nb, f)


# ── Core conversion dispatcher ──────────────────────────────────────────────

def convert_file(src: Path, dst: Path, direction: str, force: bool = False) -> bool:
    """
    Convert a single file.

    Args:
        src: Source file path
        dst: Destination file path
        direction: "to_md" or "to_nb"
        force: Overwrite existing destination

    Returns:
        True if conversion succeeded
    """
    if not src.exists():
        print(f"  ⚠️  Source not found: {src}")
        return False

    if dst.exists() and not force:
        print(f"  ⏭️  Skipping (exists): {dst.name}")
        return False

    try:
        use_jupytext = _has_jupytext()

        if direction == "to_md":
            if use_jupytext:
                _convert_with_jupytext(src, dst, "md")
            else:
                _fallback_ipynb_to_md(src, dst)
        else:  # to_nb
            if use_jupytext:
                _convert_with_jupytext(src, dst, "ipynb")
            else:
                _fallback_md_to_ipynb(src, dst)

        print(f"  ✅ {src.name} → {dst.name}")
        return True

    except Exception as e:
        print(f"  ❌ Failed {src.name}: {e}")
        return False


def discover_files(ext: str) -> list[Path]:
    """List files with given extension in CURRENT DIRECTORY only."""
    return sorted([f for f in CURRENT_DIR.iterdir() if f.is_file() and f.suffix == ext])


def main():
    parser = argparse.ArgumentParser(
        description="Convert Jupyter notebooks ↔ Markdown in the current directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python nb2md.py --to-md              # All .ipynb → .md
  python nb2md.py --to-nb --force      # All .md → .ipynb, overwrite
        """
    )
    parser.add_argument("--to-md", action="store_true", help="Convert .ipynb → .md")
    parser.add_argument("--to-nb", action="store_true", help="Convert .md → .ipynb")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")

    args = parser.parse_args()

    if not args.to_md and not args.to_nb:
        parser.print_help()
        sys.exit(1)

    if args.to_md and args.to_nb:
        print("❌ Cannot use --to-md and --to-nb together. Pick one.")
        sys.exit(1)

    # Determine direction and file sets
    if args.to_md:
        src_ext, dst_ext = ".ipynb", ".md"
        direction = "to_md"
        files = discover_files(".ipynb")
    else:
        src_ext, dst_ext = ".md", ".ipynb"
        direction = "to_nb"
        files = discover_files(".md")

    if not files:
        print(f"No *{src_ext} files found in {CURRENT_DIR}")
        sys.exit(0)

    print(f"📁 Working directory: {CURRENT_DIR}")
    print(f"🔍 Found {len(files)} {src_ext} file(s)")
    if not _has_jupytext():
        print("⚠️  jupytext not installed — using fallback converter (limited fidelity)")
    print()

    converted = 0
    skipped = 0
    failed = 0

    for src in files:
        dst = src.with_suffix(dst_ext)

        if args.dry_run:
            action = "OVERWRITE" if dst.exists() else "CREATE"
            print(f"  [DRY-RUN] {src.name} → {dst.name} ({action})")
            continue

        if convert_file(src, dst, direction, force=args.force):
            converted += 1
        elif dst.exists() and not args.force:
            skipped += 1
        else:
            failed += 1

    if not args.dry_run:
        print(f"\n📊 Done: {converted} converted, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()