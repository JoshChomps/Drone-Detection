"""
Unpack split zip archives (name.z01, name.z02, ..., name.zip) without 7-Zip.

The multi-view drone tracking datasets (03 section 2.3) ship each camera's video this way. Python's zipfile reads one
file only and refuses zip64 archives that span parts, so this reads the archive's directory itself: the parts are
one continuous stream, each member is found by its part number and offset, and its CRC is checked while it is written.

    python tools/unpack_split_zip.py <folder with split zips> <output folder>
Finds every *.zip under the folder (with its .zNN parts beside it) and writes the members under the output folder,
keeping each archive's sub-folder. An existing file of the right size is skipped.
"""
import io
import struct
import sys
import zlib
from pathlib import Path


class Parts(io.RawIOBase):
    """Several files read as one seekable stream."""

    def __init__(self, paths):
        self.files = [open(p, "rb") for p in paths]
        self.sizes = [p.stat().st_size for p in paths]
        self.starts = [sum(self.sizes[:i]) for i in range(len(paths))]
        self.total, self.pos = sum(self.sizes), 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.total + off
        return self.pos

    def readinto(self, b):
        if self.pos >= self.total:
            return 0
        i = max(k for k, s in enumerate(self.starts) if s <= self.pos)
        f = self.files[i]
        f.seek(self.pos - self.starts[i])
        n = f.readinto(memoryview(b)[:min(len(b), self.sizes[i] - (self.pos - self.starts[i]))])
        self.pos += n
        return n

    def close(self):
        for f in self.files:
            f.close()
        super().close()


def parts_of(last):
    """name.z01 ... name.zNN in order, then name.zip (the last part, which holds the directory)."""
    return sorted(last.parent.glob(last.stem + ".z[0-9]*")) + [last]


def directory(stream):
    """Members of the archive as (name, part, offset in that part, method, crc, packed size, size), read from the
    central directory at the end of the last part. Handles zip64 (a video over 4 GB), which zipfile refuses to read
    across parts."""
    tail_at = max(0, stream.total - (1 << 16) - 64)
    stream.seek(tail_at)
    tail = stream.read(stream.total - tail_at)
    e = tail.rfind(b"PK\x05\x06")
    _, cd_part, _, n, cd_size, cd_off = struct.unpack("<HHHHII", tail[e + 4:e + 20])
    loc = tail.rfind(b"PK\x06\x07", 0, e)
    if loc >= 0 and (n == 0xFFFF or cd_off == 0xFFFFFFFF or cd_part == 0xFFFF):
        part64, off64 = struct.unpack("<IQ", tail[loc + 4:loc + 16])
        stream.seek(stream.starts[part64] + off64)
        rec = stream.read(56)
        assert rec[:4] == b"PK\x06\x06", "zip64 end record not found"
        _, cd_part, _, n, cd_size, cd_off = struct.unpack("<IIQQQQ", rec[16:56])
    stream.seek(stream.starts[cd_part] + cd_off)
    cd = stream.read(cd_size)
    members, i = [], 0
    for _ in range(n):
        assert cd[i:i + 4] == b"PK\x01\x02", "central directory entry not found"
        (method, _, _, crc, csize, usize, nlen, elen, clen, part, _, _, off) = struct.unpack("<HHHIIIHHHHHII", cd[i + 10:i + 46])
        name = cd[i + 46:i + 46 + nlen].decode("utf-8", "replace")
        extra, j = cd[i + 46 + nlen:i + 46 + nlen + elen], 0
        while j + 4 <= len(extra):                 # zip64 extra field: the values that did not fit, in this order
            tag, size = struct.unpack("<HH", extra[j:j + 4])
            if tag == 1:
                v = extra[j + 4:j + 4 + size]
                if usize == 0xFFFFFFFF:
                    usize, v = struct.unpack("<Q", v[:8])[0], v[8:]
                if csize == 0xFFFFFFFF:
                    csize, v = struct.unpack("<Q", v[:8])[0], v[8:]
                if off == 0xFFFFFFFF:
                    off, v = struct.unpack("<Q", v[:8])[0], v[8:]
                if part == 0xFFFF:
                    part = struct.unpack("<I", v[:4])[0]
            j += 4 + size
        members.append((name, part, off, method, crc, csize, usize))
        i += 46 + nlen + elen + clen
    return members


def unpack(last, out):
    paths = parts_of(last)
    stream = Parts(paths)
    for name, part, off, method, crc, csize, usize in directory(stream):
        if name.endswith("/"):
            continue
        dst = out / name
        if dst.exists() and dst.stat().st_size == usize:
            print(f"  have {dst}", flush=True)
            continue
        assert method in (0, 8), f"{name}: compression method {method} not handled"
        stream.seek(stream.starts[part] + off)
        head = stream.read(30)
        assert head[:4] == b"PK\x03\x04", f"{name}: local header not found"
        nlen, elen = struct.unpack("<HH", head[26:30])
        stream.seek(stream.tell() + nlen + elen)
        dst.parent.mkdir(parents=True, exist_ok=True)
        z, got, left = zlib.decompressobj(-15) if method == 8 else None, 0, csize
        with open(dst, "wb") as f:
            while left:
                chunk = stream.read(min(left, 1 << 22))
                left -= len(chunk)
                data = z.decompress(chunk) if z else chunk
                got = zlib.crc32(data, got)
                f.write(data)
            if z:
                data = z.flush()
                got = zlib.crc32(data, got)
                f.write(data)
        if got != crc:
            dst.rename(dst.with_suffix(dst.suffix + ".badcrc"))
            raise SystemExit(f"CRC mismatch: {dst}")
        print(f"  {dst}  {usize / 1e9:.2f} GB  ({len(paths)} part{'s' if len(paths) > 1 else ''})", flush=True)


if __name__ == "__main__":
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    for last in sorted(root.rglob("*.zip")):
        print(last.relative_to(root))
        unpack(last, out / last.parent.relative_to(root).parent)
