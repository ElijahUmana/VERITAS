#!/usr/bin/env python
"""Phase-Zero #3 — snapshot_filesystem() -> fork N sandboxes.

This proves the snapshot/fork mechanism: a state becomes a frozen filesystem image,
and many child sandboxes boot from that image while diverging independently.

Verifies:
  - create sandbox, mutate its filesystem (write a marker + "expensive setup" artifact)
  - sb.snapshot_filesystem() returns a real modal.Image (a reproducible checkpoint)
  - modal.Image.from_id(img.object_id) reconstructs that image by id
  - N child sandboxes booted FROM the snapshot all already contain the artifact
    (i.e. zero redundant setup — the inheritance property)
  - children can diverge independently (each writes its own branch file) without
    affecting siblings — the basis for parallel hypothesis branches

Run (after auth):  .venv/bin/python phase-zero/modal/03_snapshot_fork.py
Env:  MODAL_VERIFY_FORKS=4  (number of child branches to fork)
"""
from __future__ import annotations
import json, os, sys, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import modal
from _common import run, log, require, ok, APP_NAME, GPU, GPU_IS_CPU

N_FORKS = int(os.environ.get("MODAL_VERIFY_FORKS", "4"))
MARKER = "/root/EXPENSIVE_SETUP_ARTIFACT.txt"
BRANCH = "/root/BRANCH.txt"
PARENT_ONLY = "/root/PARENT_AFTER_SNAPSHOT.txt"
PAYLOAD = "generation-0 built this once; every child must inherit it for free"
IDENTITY_PY = (
    "import json, os, socket; "
    "hostname = socket.gethostname(); "
    "print(json.dumps({"
    "'hostname': hostname, "
    "'container_id': os.environ.get('MODAL_CONTAINER_ID') or os.environ.get('HOSTNAME') or hostname, "
    "'pid': os.getpid()"
    "}))"
)


def sandbox_identity(sb: modal.Sandbox, label: str) -> dict[str, object]:
    p = sb.exec("python", "-c", IDENTITY_PY)
    raw = p.stdout.read().strip()
    require(p.wait() == 0 and raw, f"recorded runtime identity for {label}")
    ident = json.loads(raw)
    ident["sandbox_id"] = sb.object_id
    return ident


def main() -> None:
    app = modal.App.lookup(APP_NAME, create_if_missing=True)
    image = modal.Image.debian_slim(python_version="3.12")
    gpu = None if GPU_IS_CPU else GPU

    # --- Generation 0: build state, then freeze it -----------------------------
    log("creating gen-0 sandbox (the 'expensive setup' that we want to inherit) ...")
    parent = modal.Sandbox.create(app=app, image=image, gpu=gpu, timeout=600)
    log(f"gen-0 sandbox: {parent.object_id}")
    children = []
    try:
        parent_ident = sandbox_identity(parent, "gen-0 parent")
        log(f"gen-0 identity: host={parent_ident['hostname']} container={parent_ident['container_id']}")

        w = parent.exec("bash", "-lc", f"echo '{PAYLOAD}' > {MARKER} && wc -c {MARKER}")
        log("write artifact: " + w.stdout.read().strip()); require(w.wait() == 0, "wrote artifact in gen-0")

        log("snapshot_filesystem() — freezing the gen-0 state into a Modal Image ...")
        t_snap = time.time()
        img = parent.snapshot_filesystem()
        snap_dt = time.time() - t_snap
        img_id = img.object_id
        log(f"snapshot complete in {snap_dt:.1f}s -> image_id={img_id}")
        require(isinstance(img_id, str) and img_id.startswith("im-") and len(img_id) > len("im-"),
                "snapshot_filesystem() returned a Modal Image id with expected im- prefix")

        # Reconstruct the image purely from its id (proves it is a durable, addressable checkpoint)
        img_by_id = modal.Image.from_id(img_id)
        require(getattr(img_by_id, "object_id", None) == img_id,
                "modal.Image.from_id(snapshot_id) reconstructed the exact checkpoint id")

        parent_d = parent.exec("bash", "-lc", f"echo 'parent-after-snapshot' > {PARENT_ONLY} && cat {PARENT_ONLY}")
        parent_only = parent_d.stdout.read().strip()
        require(parent_d.wait() == 0 and parent_only == "parent-after-snapshot",
                "parent diverged after snapshot with a parent-only marker")

        # --- Generation 1: fork N children from the frozen snapshot -------------
        log(f"forking {N_FORKS} child sandboxes FROM the snapshot (parallel) ...")
        from concurrent.futures import ThreadPoolExecutor

        def boot_child(i: int):
            t0 = time.time()
            c = modal.Sandbox.create(app=app, image=img_by_id, gpu=gpu, timeout=300)
            ident = sandbox_identity(c, f"child {i}")
            # 1) inheritance: the artifact must already be present (no re-setup)
            r = c.exec("cat", MARKER)
            content = r.stdout.read().strip()
            require(r.wait() == 0, f"child {i} read inherited artifact")
            p = c.exec("bash", "-lc", f"test ! -e {PARENT_ONLY} && printf absent || cat {PARENT_ONLY}")
            parent_seen = p.stdout.read().strip()
            require(p.wait() == 0 and parent_seen == "absent",
                    f"child {i} did not inherit the parent's post-snapshot divergence")
            # 2) divergence: each child writes its own branch marker
            b = c.exec("bash", "-lc", f"echo 'branch-{i}' > {BRANCH} && cat {BRANCH}")
            branch = b.stdout.read().strip()
            require(b.wait() == 0, f"child {i} wrote branch marker")
            return {"i": i, "id": c.object_id, "inherited": content, "branch": branch,
                    "identity": ident, "boot_s": round(time.time() - t0, 1), "sb": c}

        with ThreadPoolExecutor(max_workers=N_FORKS) as ex:
            results = list(ex.map(boot_child, range(N_FORKS)))
        children = [r["sb"] for r in results]

        for r in results:
            ident = r["identity"]
            log(f"  child {r['i']}: id={r['id']} host={ident['hostname']} "
                f"container={ident['container_id']} boot={r['boot_s']}s "
                f"inherited={r['inherited']!r} branch={r['branch']!r}")

        parent_branch = parent.exec("bash", "-lc", f"test ! -e {BRANCH} && printf absent || cat {BRANCH}")
        parent_branch_seen = parent_branch.stdout.read().strip()
        require(parent_branch.wait() == 0 and parent_branch_seen == "absent",
                "parent remained isolated from child branch divergence")

        for r in results:
            rb = r["sb"].exec("cat", BRANCH)
            r["branch_after_siblings"] = rb.stdout.read().strip()
            require(rb.wait() == 0, f"child {r['i']} branch marker remained readable")

        child_ids = [r["id"] for r in results]
        child_hostnames = [r["identity"]["hostname"] for r in results]

        # Assertions: every child inherited the artifact, identities are distinct, and branches are independent
        require(len(set(child_ids)) == N_FORKS and parent.object_id not in set(child_ids),
                "child sandboxes have distinct Modal object_ids from parent and each other")
        if all(child_hostnames) and len(set(child_hostnames)) == N_FORKS \
                and parent_ident["hostname"] not in set(child_hostnames):
            ok("child sandboxes reported distinct hostnames from parent and each other")
        else:
            log("identity note: Modal reported non-unique hostnames; using Modal sandbox object_ids "
                "as the authoritative sandbox identity oracle")
        require(all(r["inherited"] == PAYLOAD for r in results),
                f"all {N_FORKS} children inherited the gen-0 artifact for free (zero re-setup)")
        require(sorted(r["branch"] for r in results) == sorted(f"branch-{i}" for i in range(N_FORKS)),
                "each child diverged independently (distinct branch markers, no cross-talk)")
        require(all(r["branch_after_siblings"] == f"branch-{r['i']}" for r in results),
                "each child retained only its own branch marker after all siblings diverged")
        ok(f"FORK PRIMITIVE VERIFIED: 1 snapshot -> {N_FORKS} inheriting, independently-diverging branches")
    finally:
        log("terminating all sandboxes (parent + children) ...")
        for c in children:
            try:
                c.terminate()
            except Exception as e:  # report each cleanup error; continue tearing down the rest
                log(f"  child terminate error (non-fatal): {e!r}")
        parent.terminate()
        log("teardown complete")


if __name__ == "__main__":
    run(main)
