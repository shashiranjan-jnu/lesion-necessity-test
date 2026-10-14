"""Data pipeline: PNG cache, subject-grouped splits, Dirichlet partitioning.

Grouping unit is the BASE PID (trailing alpha suffix stripped), since suffixed
PIDs denote repeat sessions of one subject. See A1 audit.
"""
import glob, os, re, collections
import numpy as np

CLASSES = ["glioma", "meningioma", "pituitary"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
_PAT  = re.compile(r"^(\w+?)_PID([A-Za-z0-9]+)_img(\d+)\.png$")
_BASE = re.compile(r"^(.*?\d+)[A-Z]*$")

NATIVE = 224          # stored resolution; resize at load time


def _base_pid(pid: str) -> str:
    m = _BASE.match(pid)
    return m.group(1) if m else pid


# ----------------------------------------------------------------- cache
def build_cache(root: str, out_path: str) -> str:
    """Read every PNG once, stack into arrays, save a single compressed .npz."""
    from PIL import Image

    recs = []
    for c in CLASSES:
        for f in sorted(glob.glob(f"{root}/{c}/*.png")):
            m = _PAT.match(os.path.basename(f))
            if not m:
                raise ValueError(f"unparseable filename: {f}")
            recs.append((f, CLS2IDX[c], _base_pid(m.group(2))))

    n = len(recs)
    images = np.zeros((n, NATIVE, NATIVE), dtype=np.uint8)
    labels = np.zeros(n, dtype=np.int64)
    pids   = np.empty(n, dtype=object)

    for i, (f, y, pid) in enumerate(recs):
        im = Image.open(f).convert("L")
        if im.size != (NATIVE, NATIVE):
            im = im.resize((NATIVE, NATIVE), Image.BILINEAR)
        images[i], labels[i], pids[i] = np.array(im), y, pid

    pids = np.array(pids.tolist())
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, images=images, labels=labels, pids=pids)
    print(f"cached {n} images -> {out_path} "
          f"({os.path.getsize(out_path)/1e6:.1f} MB, {len(set(pids))} subjects)")
    return out_path


def load_arrays(npz_path: str):
    d = np.load(npz_path, allow_pickle=False)
    return d["images"], d["labels"], d["pids"]


# ----------------------------------------------------------------- splits
def subject_split(pids, labels, seed: int, test_frac=0.15, val_frac=0.15):
    """Stratified subject-grouped split. Each subject has exactly one class
    (verified in A1), so we stratify by splitting within each class."""
    rng = np.random.default_rng(seed)
    subj_cls = {}
    for p, y in zip(pids, labels):
        subj_cls[p] = y

    tr, va, te = [], [], []
    for k in range(len(CLASSES)):
        s = np.array(sorted([p for p, y in subj_cls.items() if y == k]))
        rng.shuffle(s)
        n_te = max(1, int(round(len(s) * test_frac)))
        n_va = max(1, int(round(len(s) * val_frac)))
        te += list(s[:n_te])
        va += list(s[n_te:n_te + n_va])
        tr += list(s[n_te + n_va:])

    idx = lambda subs: np.where(np.isin(pids, list(subs)))[0]
    return idx(tr), idx(va), idx(te), (set(tr), set(va), set(te))


def dirichlet_partition(train_idx, pids, labels, n_clients: int,
                        alpha: float, seed: int, min_per_client=2):
    """Assign whole SUBJECTS to clients. Class proportions per client drawn
    from Dirichlet(alpha). Retries until every client holds >= min_per_client
    subjects spanning >= 2 classes."""
    rng = np.random.default_rng(seed)
    subj_cls = {}
    for i in train_idx:
        subj_cls[pids[i]] = labels[i]

    for attempt in range(200):
        buckets = [[] for _ in range(n_clients)]
        for k in range(len(CLASSES)):
            s = np.array(sorted([p for p, y in subj_cls.items() if y == k]))
            rng.shuffle(s)
            p = rng.dirichlet(np.repeat(alpha, n_clients))
            cuts = (np.cumsum(p) * len(s)).astype(int)[:-1]
            for j, part in enumerate(np.split(s, cuts)):
                buckets[j] += list(part)

        ok = all(len(b) >= min_per_client and
                 len({subj_cls[p] for p in b}) >= 2 for b in buckets)
        if ok:
            shards = [np.where(np.isin(pids, b))[0] for b in buckets]
            shards = [np.intersect1d(sh, train_idx) for sh in shards]
            return shards, buckets

    raise RuntimeError(
        f"no valid partition after 200 tries (n_clients={n_clients}, alpha={alpha}). "
        f"Lower n_clients or raise alpha.")


# ----------------------------------------------------------------- reporting
def partition_report(shards, labels):
    """Per-client class counts. Report this alongside nominal alpha in the paper."""
    print(f"{'client':>7} {'n':>6} | " + "".join(f"{c[:6]:>8}" for c in CLASSES))
    print("-" * 40)
    for j, sh in enumerate(shards):
        cnt = collections.Counter(labels[sh].tolist())
        print(f"{j:>7} {len(sh):>6} | " +
              "".join(f"{cnt.get(k,0):>8}" for k in range(len(CLASSES))))
    sizes = [len(s) for s in shards]
    print(f"  sizes: min {min(sizes)}  max {max(sizes)}  "
          f"-> worst-case q at B=8: {8/min(sizes):.4f}")


# ----------------------------------------------------------------- gate A2
def verify(npz_path: str, n_clients=5, alpha=0.5, seed=1):
    images, labels, pids = load_arrays(npz_path)
    assert len(images) == len(labels) == len(pids) == 3064, "wrong item count"
    assert images.dtype == np.uint8 and images.shape[1:] == (NATIVE, NATIVE)

    tr, va, te, (Str, Sva, Ste) = subject_split(pids, labels, seed)
    assert not (Str & Sva) and not (Str & Ste) and not (Sva & Ste), \
        "SUBJECT LEAKAGE ACROSS SPLITS"
    assert len(tr) + len(va) + len(te) == 3064, "split does not cover all items"
    print(f"subjects  train {len(Str)}  val {len(Sva)}  test {len(Ste)}")
    print(f"slices    train {len(tr)}  val {len(va)}  test {len(te)}")

    shards, buckets = dirichlet_partition(tr, pids, labels, n_clients, alpha, seed)
    allsub = [p for b in buckets for p in b]
    assert len(allsub) == len(set(allsub)), "subject in two clients"
    assert sum(len(s) for s in shards) == len(tr), "shards do not cover train set"
    print(f"\npartition N={n_clients}, alpha={alpha}:")
    partition_report(shards, labels)
    print("\nA2 GATE: PASS")





# ----------------------------------------------------------------- DP batching
Q_TARGET = 0.01
B_MIN    = 1

def client_batch_size(n_j: int, q: float = Q_TARGET, b_min: int = B_MIN) -> int:
    """Per-client batch size holding the sampling rate ~constant across shards.

    Under Dirichlet partitioning client sizes vary ~10x. With fixed B the rate
    q_j = B/n_j varies too, and since we report max_j eps_j the smallest silo
    dominates the guarantee. Fixing q instead equalises the per-client cost.
    """
    return max(b_min, int(round(q * n_j)))


def worst_case_q(shards, q=Q_TARGET, b_min=B_MIN) -> float:
    return max(client_batch_size(len(s), q, b_min) / len(s) for s in shards)
