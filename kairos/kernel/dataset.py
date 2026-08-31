"""Columnar KuaiRand loader with an enforced hidden-test discipline.

Two invariants this module exists to guarantee:

1. ROW ORDER is byte-identical to the official `data.load()`, because submission alignment
   is checked positionally by `submit.py` (row_id is a positional index, and
   (user_id, video_id) is NOT unique - 3.06% of test rows are repeated pairs).

2. TEST LABELS ARE UNREACHABLE from feature-building code.  In our sandbox we happen to
   possess them; in the real evaluation we do not.  Any pipeline that silently reads them
   would score brilliantly here and collapse on the real hidden test.  So labels live
   behind `LabelVault`, which serves them only up to a declared horizon date, and the
   sealed remainder is reachable only through `Scorer.score()`, which writes an audit line
   every time it is called.  A pipeline is only trustworthy if that audit log is short.
"""
import json, os, time
import numpy as np

# KuaiRand ships three variants sharing one schema and one calendar (20220408-20220508).
# Selecting by env var keeps every downstream `load()` call site untouched, so a transfer
# run exercises the SAME code path as the tuned run - which is the point of the transfer.
VARIANT = os.environ.get('KAIROS_VARIANT', 'pure').lower()
if VARIANT not in ('pure', '1k', '27k'):
    raise ValueError(f"KAIROS_VARIANT='{VARIANT}' not in pure|1k|27k")
_SUF = {'pure': 'pure', '1k': '1k', '27k': '27k'}[VARIANT]
_DIR = {'pure': 'KuaiRand-Pure', '1k': 'KuaiRand-1K', '27k': 'KuaiRand-27K'}[VARIANT]

DATA_DIR = f'./{_DIR}/data'
# separate cache per variant: the columnar cache is keyed by nothing but its path, so
# sharing one directory across variants would serve Pure's columns for a 1k run
CACHE_DIR = './runs/cache' if VARIANT == 'pure' else f'./runs/cache_{_SUF}'
STANDARD_LOGS = (f'log_standard_4_08_to_4_21_{_SUF}.csv',
                 f'log_standard_4_22_to_5_08_{_SUF}.csv')
RANDOM_LOG = f'log_random_4_22_to_5_08_{_SUF}.csv'



def load_cached(path, n_expected, what='signal'):
    """Load a cached per-row array, but ONLY if its length matches the current dataset.

    Every cache in this project is a bare .npy keyed by a path string. Four separate times
    a path was keyed by something that did not distinguish the dataset - by nothing, then
    by fold name - and a run silently loaded an array built for a DIFFERENT variant.
    np.load is happy to return it; the mismatch only surfaces later as a shape error in
    unrelated code, or worse, not at all where lengths coincidentally align.

    Path discipline alone clearly does not hold: the same mistake recurred at four sites in
    one afternoon. A length check at the point of load is the invariant that actually binds,
    because it does not depend on remembering anything.

    Returns None when there is nothing valid to load, so callers rebuild.
    """
    if not os.path.exists(path):
        return None
    arr = np.load(path, allow_pickle=False)
    if arr.shape[0] != n_expected:
        raise LeakageError(
            f"cached {what} at {path} has {arr.shape[0]:,} rows but this dataset has "
            f"{n_expected:,}. This cache was built for a different variant or split. "
            f"Delete it or key its path by variant (see variant_path).")
    return arr


# Whether a model that scores TEST rows may also be fit on VALIDATION rows.
#
# The rules are genuinely ambiguous here and we do not get to resolve them by preference:
#   - FAQ 2.9.2 states "training data is the train split only: date 20220408-20220421".
#   - But 2.3 lists out-of-scope as "no hidden-test access during development (train +
#     validation only)", 2.4 says "teams develop on train + validation only", and 2.9.3's
#     disqualification clause is specifically about touching TEST labels.
#   - Decisively, 2.9.2's own RATIONALE does not reach validation: it forbids log_random
#     because that file "covers both the validation and the test window, so training on it
#     injects in-period information about the scored rows and breaks the temporal split".
#     Validation ends 20220421-28, entirely BEFORE the test window opens on 20220429, so
#     fitting on it injects no in-period information and breaks no temporal ordering.
#
# So this is a switch, not a silent choice, and it defaults to the strict reading. The
# permissive setting is worth a replicated +0.002 (exp22, two independent backtest folds).
# Whichever is submitted, the run log records which was used.
STRICT_TRAIN_SPLIT = os.environ.get('KAIROS_STRICT_TRAIN_SPLIT', '1') != '0'


def train_end(fold_name='official'):
    """Last date whose rows may be used as MODEL TRAINING data for this fold.

    Organizer FAQ 2.9.2: "training data is the train split only: date 20220408-20220421".
    Validation is for tuning and selection, and its labels may inform FEATURE statistics
    (FAQ 2.2 permits "the training split and the public validation feedback") - but no
    model's loss may see a validation row.

    Window horizons and training cut-offs are therefore different quantities and must not
    be conflated. A frozen window over the test period legitimately aggregates labels up to
    20220428; a model scoring that window may still only FIT on rows up to 20220421.
    """
    if not STRICT_TRAIN_SPLIT:
        # Permissive reading: a model scoring the test window may also fit on validation,
        # which ends before that window opens. Never reaches a test label either way.
        return int(FOLDS[fold_name]['valid'][1])
    return int(FOLDS[fold_name]['train'][1])

def variant_path(path):
    """Suffix a cache path with the active variant, so no two variants can share a cache.

    The columnar cache was made variant-aware; every DERIVED-signal cache (fm_signal,
    refit, din, mf, cf, expert, aux) was not, which is the same bug one layer down. Those
    paths are fixed strings, so a 1k run would np.load Pure's 1,436,609-row signal into an
    11,713,045-row problem - a length mismatch at best, and silent misalignment wherever
    the length happens to be tolerated. Pure keeps its existing paths untouched so no
    cached work is invalidated.
    """
    if VARIANT == 'pure':
        return path
    root, ext = os.path.splitext(path)
    return f'{root}_{_SUF}{ext}'

LABEL = 'long_view'
# every feedback signal in the log; `long_view` is scored, the rest are auxiliary targets
FEEDBACK = ['is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward', 'is_hate',
            'long_view', 'is_profile_enter']
CONTINUOUS = ['play_time_ms', 'duration_ms', 'profile_stay_time', 'comment_stay_time']
LOG_COLS = (['user_id', 'video_id', 'date', 'hourmin', 'time_ms']
            + FEEDBACK[:-2] + ['long_view'] + CONTINUOUS + ['is_profile_enter', 'is_rand', 'tab'])

# Official split (pinned by the starter kit; do not change).
OFFICIAL = {'train': (20220408, 20220421), 'valid': (20220422, 20220428),
            'test': (20220429, 20220508)}

# Backtest folds live ENTIRELY inside the public-label region (<= 20220428) so that
# simulating the whole select-then-submit procedure never touches hidden-test labels.
# These are what let us measure the validation->test selection gap honestly.
FOLDS = {
    # Same calendar on every variant. Only Pure's test labels are withheld by the
    # organizers; on 1k/27k the whole period is public, so the test window is scoreable
    # directly - which is what makes those variants usable as a val->test generalisation
    # probe with no submission budget.
    'official':   {'train': (20220408, 20220421), 'valid': (20220422, 20220428),
                   'test': (20220429, 20220508), 'sealed': VARIANT == 'pure'},
    # Backtest folds live ENTIRELY inside the public-label region (<= 20220428).  Their
    # windows are chosen so that valid+test both sit in the SPARSE logging regime
    # (~2 impressions/user/day, as the official valid/test do) rather than straddling the
    # Apr-17 density collapse - otherwise they would simulate the wrong problem.
    'backtest_a': {'train': (20220408, 20220417), 'valid': (20220418, 20220422),
                   'test': (20220423, 20220428), 'sealed': False},
    'backtest_b': {'train': (20220408, 20220415), 'valid': (20220416, 20220420),
                   'test': (20220421, 20220426), 'sealed': False},
    'backtest_c': {'train': (20220408, 20220419), 'valid': (20220420, 20220424),
                   'test': (20220425, 20220428), 'sealed': False},
}


class LeakageError(RuntimeError):
    """Raised when code reaches for a label it is not entitled to see."""


# --------------------------------------------------------------------------- cache build
def _read_logs(data_dir, files):
    import pandas as pd
    frames = []
    for f in files:
        df = pd.read_csv(os.path.join(data_dir, f))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)     # order preserved: file1 then file2


def build_cache(data_dir=DATA_DIR, cache_dir=CACHE_DIR, force=False):
    """Parse CSVs once into .npy columns.  Row order == official data.load() order."""
    os.makedirs(cache_dir, exist_ok=True)
    stamp = os.path.join(cache_dir, 'MANIFEST.json')
    if os.path.exists(stamp) and not force:
        return json.load(open(stamp))
    import pandas as pd
    t0 = time.time()

    log = _read_logs(data_dir, STANDARD_LOGS)
    manifest = {'n_rows': int(len(log)), 'built_at': time.strftime('%Y-%m-%d %H:%M:%S')}

    # video side
    vb = pd.read_csv(os.path.join(data_dir, f'video_features_basic_{_SUF}.csv'))
    uf = pd.read_csv(os.path.join(data_dir, f'user_features_{_SUF}.csv'))
    # video_features_statistic_* holds platform-wide counters with no timestamp, so it is
    # excluded on temporal-validity grounds and nothing reads it back. On 1k/27k it is
    # multi-GB, so parsing it would cost minutes to build a cache no code ever opens.
    _vs_path = os.path.join(data_dir, f'video_features_statistic_{_SUF}.csv')
    vs = (pd.read_csv(_vs_path) if VARIANT == 'pure' else
          pd.DataFrame({'video_id': vb['video_id']}))

    def save(name, arr):
        np.save(os.path.join(cache_dir, name + '.npy'), arr)

    import pandas.api.types as pdt
    for c in log.columns:
        col = log[c]
        if not pdt.is_numeric_dtype(col):
            codes, uniques = pd.factorize(col, sort=True)
            save('log_' + c, codes.astype(np.int32))
            save('vocab_' + c, np.asarray(uniques, dtype=object))
        else:
            arr = col.to_numpy()
            if np.issubdtype(arr.dtype, np.integer):
                arr = arr.astype(np.int64 if arr.max() > 2**31 else np.int32)
            else:
                arr = arr.astype(np.float32)
            save('log_' + c, arr)
    manifest['log_columns'] = list(log.columns)

    # Side tables are read back by POSITION (arr[video_id]), so position must equal id.
    # That holds on Pure by luck, not by contract: 1k lists 4,371,868 videos whose ids run
    # to 4,371,899 - 32 gaps. Sorting alone would shift every row after the first gap and
    # bind video attributes to the wrong video silently. Reindexing onto the full 0..max
    # range makes position == id true by construction; missing ids become NaN, which is a
    # visible absence rather than another video's value. No-op on Pure.
    def _align(df, key):
        df = df.drop_duplicates(key).set_index(key).sort_index()
        full = df.reindex(np.arange(0, int(df.index.max()) + 1))
        return full.reset_index()
    vb = _align(vb, 'video_id')
    vs = _align(vs, 'video_id')
    uf = _align(uf, 'user_id')
    for tag, df in (('vb', vb), ('vs', vs), ('uf', uf)):
        for c in df.columns:
            col = df[c]
            if not pdt.is_numeric_dtype(col):
                codes, uniques = pd.factorize(col, sort=True)
                save(f'{tag}_{c}', codes.astype(np.int32))
            else:
                save(f'{tag}_{c}', col.to_numpy().astype(np.float32))
        manifest[f'{tag}_columns'] = list(df.columns)

    manifest['build_seconds'] = round(time.time() - t0, 1)
    json.dump(manifest, open(stamp, 'w'), indent=2)
    return manifest


# --------------------------------------------------------------------------- vault
class LabelVault:
    """Serves the scored label only for rows dated <= horizon.

    Rows beyond the horizon return the sentinel -1.  This is deliberate: a feature builder
    that ignores the horizon does not get a silent leak, it gets an obviously poisoned
    value, and the leakage probe in kairos.kernel.probes will catch it.
    """
    __slots__ = ('_y', '_dates', 'horizon', '_reads')

    def __init__(self, y, dates, horizon):
        self._y, self._dates, self.horizon, self._reads = y, dates, int(horizon), 0

    def visible(self, idx=None):
        d = self._dates if idx is None else self._dates[idx]
        y = self._y if idx is None else self._y[idx]
        self._reads += 1
        return np.where(d <= self.horizon, y, np.int8(-1))

    def require_within(self, idx, what=''):
        """Assert every row in idx is inside the horizon; use before any label aggregation."""
        bad = int((self._dates[idx] > self.horizon).sum())
        if bad:
            raise LeakageError(
                f"{what}: {bad} of {len(idx)} rows are dated after the label horizon "
                f"{self.horizon}. Labels past the horizon do not exist at inference time.")
        return self._y[idx]


class Scorer:
    """The only route to sealed labels. Every call is audited."""

    def __init__(self, group_ids, y, name, audit_path='./runs/scorer_audit.log'):
        self._gid, self._y, self.name, self.audit_path = group_ids, y, name, audit_path
        self.calls = 0

    def score(self, scores, reason=''):
        from kairos.kernel.fastmetrics import fast_evaluate
        self.calls += 1
        m = fast_evaluate(self._gid, self._y, np.asarray(scores, dtype=np.float64))
        os.makedirs(os.path.dirname(self.audit_path), exist_ok=True)
        with open(self.audit_path, 'a') as fh:
            fh.write(json.dumps({'t': time.strftime('%H:%M:%S'), 'split': self.name,
                                 'call': self.calls, 'reason': reason,
                                 'primary': round(m['primary'], 6)}) + '\n')
        return m


# --------------------------------------------------------------------------- data object
class Fold:
    """One temporal experiment: train / valid windows with visible labels, sealed test."""

    def __init__(self, data, name):
        spec = FOLDS[name]
        self.name, self.spec = name, spec
        self.data = data
        d = data.date
        self.idx = {}
        for part in ('train', 'valid', 'test'):
            lo, hi = spec[part]
            self.idx[part] = np.flatnonzero((d >= lo) & (d <= hi))
        # labels are visible up to the END of validation; nothing later
        self.horizon = spec['valid'][1]
        self.vault = LabelVault(data.y_raw, d, self.horizon)
        self.scorers = {}
        for part in ('valid', 'test'):
            i = self.idx[part]
            self.scorers[part] = Scorer(data.user_id[i], data.y_raw[i], f'{name}/{part}')

    def n(self, part):
        return len(self.idx[part])

    def groups(self, part):
        return self.data.user_id[self.idx[part]]

    def labels(self, part):
        """Labels for a split. Refuses the sealed test split."""
        if self.spec['sealed'] and part == 'test':
            raise LeakageError(
                "test labels are sealed for the 'official' fold - use fold.scorers['test']"
                ".score(...) which is audited, or run on a backtest_* fold instead.")
        return self.data.y_raw[self.idx[part]]

    def __repr__(self):
        return (f"<Fold {self.name} train={self.n('train'):,} valid={self.n('valid'):,} "
                f"test={self.n('test'):,} horizon={self.horizon} sealed={self.spec['sealed']}>")


class Data:
    """All log columns as aligned numpy arrays, in official row order."""

    def __init__(self, cache_dir=CACHE_DIR):
        self.cache_dir = cache_dir
        self.manifest = json.load(open(os.path.join(cache_dir, 'MANIFEST.json')))
        self._cache = {}
        self.user_id = self.col('user_id')
        self.video_id = self.col('video_id')
        self.date = self.col('date')
        self.time_ms = self.col('time_ms')
        self.y_raw = (self.col('long_view') != 0).astype(np.int8)
        self.n = len(self.user_id)
        # strict causal order: ties in time_ms broken by original row index
        self.time_order = np.lexsort((np.arange(self.n), self.time_ms))

    def col(self, name, table='log'):
        key = f'{table}_{name}'
        if key not in self._cache:
            p = os.path.join(self.cache_dir, key + '.npy')
            if not os.path.exists(p):
                raise KeyError(f"no cached column {key}")
            self._cache[key] = np.load(p, allow_pickle=name in ('upload_dt',))
        return self._cache[key]

    def fold(self, name='official'):
        return Fold(self, name)


def load(cache_dir=CACHE_DIR, data_dir=DATA_DIR):
    build_cache(data_dir, cache_dir)
    return Data(cache_dir)
