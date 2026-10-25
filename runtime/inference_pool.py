"""Request ownership and demand checks for bounded warm-model workers."""
from pathlib import Path
import fcntl
import json
import os
from shared_gpu import identity, atomic


class YieldForReview(Exception):
    pass


def read(path):
    return json.loads(Path(path).read_text())


def live_requests(folder):
    result = []
    for name in ("queue", "processing"):
        for path in (Path(folder) / name).glob("*.json"):
            try:
                value = read(path)
                if identity(value["pid"]) == value["identity"] and not Path(value["response"]).exists():
                    result.append(value)
            except FileNotFoundError:
                pass
    return result


def review_demand(runtime):
    folder = Path(runtime) / "review"
    if live_requests(folder):
        return True
    try:
        lease = read(folder / "gpu_stage_lease.json")
        return lease.get("state") == "held" and identity(lease["pid"]) == lease["identity"]
    except FileNotFoundError:
        return False


def slot_folder(runtime, kind, slot=0):
    base = Path(runtime) / kind
    return base if not slot else base / ("worker_" + str(slot))


def claim(folder, slot, reply):
    """Claim FIFO exactly once; recover only requests owned by a dead worker."""
    folder = Path(folder)
    queue, working = folder / "queue", folder / "processing"
    with (folder / "dispatch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for path in working.glob("*.json"):
            try:
                value = read(path)
            except FileNotFoundError:
                continue  # The owning worker just committed and removed it.
            owner = value.get("service_pid")
            if owner and identity(owner) == value.get("service_identity"):
                continue
            # Inference has no side effects until response.json commits. A dead
            # worker's uncommitted request can safely return to its FIFO place.
            if Path(value["response"]).exists() or identity(value["pid"]) != value["identity"]:
                path.unlink(missing_ok=True)
            else:
                os.replace(path, queue / path.name)
        for path in sorted(queue.glob("*.json")):
            value = read(path)
            if Path(value["response"]).exists() or identity(value["pid"]) != value["identity"]:
                path.unlink()
                continue
            dest = working / path.name
            # Both publication and rename occur under the claim lock. Another
            # worker never mistakes a live claim for an interrupted request.
            value.update(service_pid=os.getpid(), service_identity=identity(os.getpid()), service_slot=slot)
            atomic(path, value)
            os.replace(path, dest)
            return dest, value
    return None


def return_claim(folder, path):
    with (Path(folder) / "dispatch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        os.replace(path, Path(folder) / "queue" / path.name)
