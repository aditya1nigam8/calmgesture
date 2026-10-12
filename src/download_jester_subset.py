"""Read official split TGZ parts over HTTP, saving only selected clip frames.

The whole compressed stream must be transferred (about 23 GB); only the selected
JPGs are stored. Each byte range is checked and retried before decompression.
"""
import argparse
import io
import re
import time
from urllib.request import Request, urlopen

from extract_jester_subset import extract_subset
from prepare_jester import read_manifest

BASE = "https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/dataset/AIDataset/Jester/"
URLS = [BASE + f"20bnjester-v1-{i:02d}" for i in range(3)]


def chunks(urls, chunk_size=8 * 1024 * 1024):
    transferred = 0
    for part, url in enumerate(urls, 1):
        offset, total = 0, None
        while total is None or offset < total:
            end = offset + chunk_size - 1
            if total is not None:
                end = min(end, total - 1)
            for attempt in range(20):
                try:
                    request = Request(url, headers={"Range": f"bytes={offset}-{end}"})
                    with urlopen(request, timeout=60) as response:
                        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                        if response.status != 206 or not match:
                            raise ValueError("Server did not honor partial download")
                        start, stop, size = map(int, match.groups())
                        if start != offset or stop != min(end, size - 1) or (total is not None and size != total):
                            raise ValueError("Unexpected archive byte range")
                        content = response.read(chunk_size + 1)
                        if len(content) != stop - start + 1:
                            raise ValueError("Truncated archive byte range")
                    total = size
                    break
                except (OSError, ValueError) as error:
                    if attempt == 19:
                        raise
                    print(f"Retrying part {part} at byte {offset}: {error}", flush=True)
                    time.sleep(min(2 ** attempt, 30))
            offset += len(content)
            transferred += len(content)
            if offset == total or offset % (128 * 1024 * 1024) == 0:
                print(f"Part {part}/3: {offset / 1e9:.2f}/{total / 1e9:.2f} GB; transferred {transferred / 1e9:.2f} GB", flush=True)
            yield content


class ChunkReader(io.RawIOBase):
    def __init__(self, source):
        self.source = iter(source)
        self.pending = memoryview(b"")

    def readable(self):
        return True

    def readinto(self, buffer):
        if not self.pending:
            self.pending = memoryview(next(self.source, b""))
        count = min(len(buffer), len(self.pending))
        buffer[:count] = self.pending[:count]
        self.pending = self.pending[count:]
        return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/jester/subset.csv")
    parser.add_argument("--output", default="data/jester/frames")
    parser.add_argument("--resume", action="store_true", help="Verify/reuse frames; compressed stream restarts from the beginning")
    args = parser.parse_args()
    rows = read_manifest(args.manifest)
    with io.BufferedReader(ChunkReader(chunks(URLS))) as stream:
        counts = extract_subset(stream, rows, args.output, args.resume)
    print(f"Complete: {len(counts)} selected clips, {sum(counts.values())} JPG frames", flush=True)


if __name__ == "__main__":
    main()
