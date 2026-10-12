"""Stream a Jester tar archive, retaining only JPG frames in the chosen manifest.

Use --archive - for stdin (e.g. concatenated official TGZ parts). This reduces
stored data, not network traffic: a gzip stream still needs sequential reading.
"""
import argparse
import json
import sys
import tarfile
from pathlib import Path, PurePosixPath

from prepare_jester import read_manifest


def extract_subset(stream, rows, output, resume=False):
    wanted = {row["clip_id"] for row in rows}
    counts = dict.fromkeys(wanted, 0)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=resume)
    state_path = output / "extraction.json"
    selection = sorted(wanted, key=int)
    if state_path.exists():
        previous = json.loads(state_path.read_text())
        if previous["clip_ids"] != selection:
            raise ValueError("Output directory belongs to a different clip selection")
    state_path.write_text(json.dumps({"status": "incomplete", "clip_ids": selection}) + "\n")
    with tarfile.open(fileobj=stream, mode="r|*") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            if not member.isfile() or len(path.parts) < 2 or ".." in path.parts or path.is_absolute():
                continue
            clip_id, filename = path.parts[-2:]
            if clip_id not in wanted or path.suffix.lower() != ".jpg" or not path.stem.isdigit():
                continue
            if member.size > 20 * 1024 * 1024:
                raise ValueError(f"Unexpectedly large image: {member.name}")
            target = output / clip_id / filename
            target.parent.mkdir(exist_ok=True)
            with archive.extractfile(member) as source:
                content = source.read()
            if target.exists():
                if not resume or target.read_bytes() != content:
                    raise ValueError(f"Existing frame differs from archive: {target}")
            else:
                partial = target.with_suffix(".jpg.partial")
                with partial.open("wb") as destination:
                    destination.write(content)
                partial.replace(target)
            counts[clip_id] += 1
    missing = [clip for clip, count in counts.items() if count == 0]
    if missing:
        raise ValueError(f"Archive missing {len(missing)} selected clips; first IDs: {sorted(missing)[:10]}. Partial output is retained.")
    state_path.write_text(json.dumps({"status": "complete", "clip_ids": selection, "frame_counts": counts}) + "\n")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--archive", required=True, help="Tar/tar.gz file, or - for stdin")
    parser.add_argument("--output", type=Path, default=Path("data/jester/frames"))
    parser.add_argument("--resume", action="store_true", help="Verify and reuse identical existing frames")
    args = parser.parse_args()
    rows = read_manifest(args.manifest)
    if args.archive == "-":
        counts = extract_subset(sys.stdin.buffer, rows, args.output, args.resume)
    else:
        with open(args.archive, "rb") as handle:
            counts = extract_subset(handle, rows, args.output, args.resume)
    print(f"Extracted {sum(counts.values())} frames from {len(counts)} selected clips")


if __name__ == "__main__":
    main()
