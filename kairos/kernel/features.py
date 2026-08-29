"""Categorical field encoding with a train-only vocabulary.

Starts at exact parity with the official baseline's 5 fields so that any measured change
is attributable to the thing we changed and not to a different feature set.  New fields
plug in through `FieldSpec`, which is the surface the agent is allowed to write against.
"""
import numpy as np

BASELINE_FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket']


class FieldSpec:
    """One categorical field: a name plus a function producing its raw values per row."""

    def __init__(self, name, fn):
        self.name, self.fn = name, fn


def default_specs(data):
    """The official baseline's five fields, reproduced exactly."""
    vb_author = data.col('author_id', 'vb')          # indexed by dense video_id

    def author(idx):
        v = data.video_id[idx]
        a = np.full(len(idx), -1, dtype=np.int64)
        ok = v < len(vb_author)
        a[ok] = vb_author[v[ok]].astype(np.int64)
        return a

    return [
        FieldSpec('user_id',    lambda i: data.user_id[i].astype(np.int64)),
        FieldSpec('video_id',   lambda i: data.video_id[i].astype(np.int64)),
        FieldSpec('author_id',  author),
        FieldSpec('tab',        lambda i: data.col('tab')[i].astype(np.int64)),
        FieldSpec('dur_bucket', None),               # filled in by Encoder (needs train quantiles)
    ]


class Encoder:
    """Builds train-only vocabularies, maps every split into a dense int32 matrix.

    Mirrors the official data.encode(): unseen values fall into a per-field UNK slot, and
    field id spaces are concatenated with offsets so a single embedding table serves all.
    """

    def __init__(self, data, specs=None, n_dur_buckets=10):
        self.data = data
        self.specs = specs if specs is not None else default_specs(data)
        self.n_dur_buckets = n_dur_buckets
        self.fitted = False

    def _dur_bucket(self, idx):
        dur = self.data.col('duration_ms')[idx].astype(np.float64)
        return np.searchsorted(self.edges, dur).astype(np.int64)

    def _raw(self, idx):
        out = []
        for sp in self.specs:
            out.append(self._dur_bucket(idx) if sp.name == 'dur_bucket' else sp.fn(idx))
        return out

    def fit(self, train_idx):
        dur = self.data.col('duration_ms')[train_idx].astype(np.float64)
        self.edges = np.quantile(dur, np.linspace(0, 1, self.n_dur_buckets + 1)[1:-1])
        self.vocabs, self.field_dims = [], []
        for col in self._raw(train_idx):
            uniq = np.unique(col)
            self.vocabs.append(uniq)
            self.field_dims.append(len(uniq) + 1)     # +1 UNK slot at the end
        self.offsets = np.cumsum([0] + self.field_dims[:-1]).astype(np.int64)
        self.dim = int(sum(self.field_dims))
        self.fitted = True
        return self

    def transform(self, idx):
        assert self.fitted, "call fit(train_idx) first"
        X = np.empty((len(idx), len(self.specs)), dtype=np.int32)
        for f, col in enumerate(self._raw(idx)):
            pos = np.searchsorted(self.vocabs[f], col)
            pos = np.clip(pos, 0, len(self.vocabs[f]) - 1)
            hit = self.vocabs[f][pos] == col
            code = np.where(hit, pos, len(self.vocabs[f]))     # miss -> UNK slot
            X[:, f] = (code + self.offsets[f]).astype(np.int32)
        return X

    def names(self):
        return [sp.name for sp in self.specs]
