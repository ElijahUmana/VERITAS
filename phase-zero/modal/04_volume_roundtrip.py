#!/usr/bin/env python
"""Phase-Zero #4 — modal.Volume round-trip.

A shared Volume lets one worker commit while another reloads and reads. This
proves durable shared state across Modal compute.

Verifies:
  - Volume.from_name(create_if_missing=True) creates/attaches a named volume
  - a function writes to the mount and .commit()s
  - a reader function .reload()s and reads the committed bytes back
  - writer/reader runtime identities are returned and compared; if Modal reuses
    a container, the run is qualified instead of claiming cross-container proof
  - last-write-wins semantics observed (we overwrite and re-read)

Run (after auth):  .venv/bin/modal run phase-zero/modal/04_volume_roundtrip.py
"""
from __future__ import annotations
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import modal
from _common import APP_NAME, VERIFY_VOLUME

app = modal.App(f"{APP_NAME}-vol")
vol = modal.Volume.from_name(VERIFY_VOLUME, create_if_missing=True)
MOUNT = "/data"
KEY = f"{MOUNT}/handshake.txt"
COMMON_PY = pathlib.Path(__file__).resolve().with_name("_common.py")
image = modal.Image.debian_slim(python_version="3.12").add_local_file(
    COMMON_PY, "/root/_common.py", copy=True
)


def runtime_identity(role: str) -> dict[str, object]:
    import os, socket
    hostname = socket.gethostname()
    return {
        "role": role,
        "hostname": hostname,
        "container_id": os.environ.get("MODAL_CONTAINER_ID") or os.environ.get("HOSTNAME") or hostname,
        "pid": os.getpid(),
    }


@app.function(image=image, volumes={MOUNT: vol})
def writer(token: str) -> dict[str, object]:
    import os
    os.makedirs(MOUNT, exist_ok=True)
    with open(KEY, "w") as f:
        f.write(token)
    vol.commit()  # without this, the write is lost on container exit
    info = runtime_identity("writer")
    info.update({"bytes": len(token), "path": KEY})
    return info


@app.function(image=image, volumes={MOUNT: vol})
def reader() -> dict[str, object]:
    vol.reload()  # pull the latest committed state from the volume
    with open(KEY) as f:
        content = f.read()
    info = runtime_identity("reader")
    info.update({"content": content, "path": KEY})
    return info


def identity_label(info: dict[str, object]) -> str:
    return f"{info['role']} host={info['hostname']} container={info['container_id']} pid={info['pid']}"


def assert_identity(label: str, info: dict[str, object]) -> None:
    assert info.get("hostname"), f"{label} did not return a hostname"
    assert info.get("container_id"), f"{label} did not return a container_id/hostname identity"
    assert isinstance(info.get("pid"), int) and info["pid"] > 0, f"{label} did not return a valid pid"


def compare_container_identity(label: str, write_info: dict[str, object], read_info: dict[str, object]) -> bool:
    assert_identity(f"{label} writer", write_info)
    assert_identity(f"{label} reader", read_info)
    same_container = (
        write_info["container_id"] == read_info["container_id"]
        or write_info["hostname"] == read_info["hostname"]
    )
    if same_container:
        print(
            f"  {label}: Modal reported the same writer/reader container identity; "
            "qualifying this run as commit/reload proof, not cross-container proof",
            flush=True,
        )
        return False
    print(f"  {label}: separate containers verified ({identity_label(write_info)} != {identity_label(read_info)})",
          flush=True)
    return True


@app.local_entrypoint()
def main() -> None:
    token = f"veritas-{int(time.time())}"
    print(f"[04_volume] writer.remote({token!r}) ...", flush=True)
    wrote = writer.remote(token)
    print("  " + identity_label(wrote) + f" wrote+committed {wrote['bytes']} bytes to {wrote['path']}", flush=True)

    print("[04_volume] reader.remote() ...", flush=True)
    read = reader.remote()
    got = read["content"]
    print(f"  read back: {got!r} from {identity_label(read)}", flush=True)
    assert got == token, f"volume round-trip mismatch: wrote {token!r}, read {got!r}"
    separation_observed = [compare_container_identity("round 1", wrote, read)]

    # last-write-wins overwrite
    token2 = token + "-v2"
    wrote2 = writer.remote(token2)
    read2 = reader.remote()
    got2 = read2["content"]
    assert got2 == token2, f"overwrite round-trip mismatch: wrote {token2!r}, read {got2!r}"
    separation_observed.append(compare_container_identity("round 2", wrote2, read2))

    separation_note = (
        "writer/reader container separation observed in both rounds"
        if all(separation_observed)
        else "Modal reused a writer/reader container in at least one round; no cross-container claim made"
    )
    print(
        f"\033[32mPASS\033[0m  Volume round-trip verified live "
        f"(write->commit->reload->read, twice; {separation_note})",
        flush=True,
    )
