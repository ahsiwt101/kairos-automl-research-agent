"""Hypothesis + code generation.

Token efficiency is a scored criterion here (Feasibility, 15%, ranked among submissions
that beat the baseline), so the prompt is built from a DIGEST - metrics, slice headroom,
inversion attribution, the ledger's family track record - rather than raw logs.  The agent
is meant to be cheap because it is well-informed, not cheap because it thinks less.

Every proposal must commit to a falsifiable prediction about which diagnostic slice will
move and in which direction.  That prediction is scored afterwards, which is what lets the
scheduler tell a family that understands the problem from one that got lucky.
"""
import json, os, re, textwrap

SYSTEM = """You are an ML research agent improving a recommender ranking pipeline.

TASK. KuaiRand-Pure, within-user ranking of logged impressions. Label `long_view` (binary).
Metric: primary = mean(GAUC, nDCG@5). GAUC is per-user AUC weighted by each user's positive
count, excluding all-positive and all-negative users. nDCG@5 scores zero-positive users as 0.
Official FM baseline: validation 0.6016, hidden test 0.5946. Oracle ceiling 0.8645.
Splits: train 20220408-21, valid 20220422-28, hidden test 20220429-0508.

WHAT IS ALREADY KNOWN (do not re-derive):
- Extra static features and extra embedding capacity give nothing (organiser-tested).
- Changing the loss (BPR / listwise / LambdaRank / GAUC-exact pairwise) gives nothing
  measurable; all within seed noise of pointwise BCE.
- Per-seed std is 0.0008, so any difference below ~0.002 needs multiple seeds to claim.
- Within-user ranking is invariant to per-user constants: a feature that is constant
  across a user's evaluation list CANNOT change the metric.
- The logging density collapses 5x mid-window; valid and test are both in the sparse regime.
- Your candidate is trained by LightGBM on the matrix you return, so you cannot rediscover
  the FM's ID crosses. `ctx.baseline_score` gives you them directly; a matrix WITHOUT it
  will almost certainly score below the baseline and be rejected.
- `ctx.mf_factors`, `ctx.auxiliary_signal`, and `ctx.cf_score` are DIFFERENT signals from
  `ctx.baseline_score` and from each other (collaborative filtering, auxiliary feedback,
  and behavioural similarity respectively) - combining several is more likely to help than
  any single one, but you decide the composition.

YOU WRITE PYTHON. Define exactly one function `build(ctx)` returning (X, names) or
(X, names, train_cfg). X must be float32 with EXACTLY ctx.data.n rows, aligned to all log
rows in data order. Write direct code - the API below is exact, so do not add defensive
fallbacks or try/except probing.

ctx.data.n                int, number of log rows
ctx.data.user_id          int32 (n,)        ctx.data.video_id   int32 (n,)
ctx.data.date             int32 (n,) yyyymmdd
ctx.data.time_ms          int64 (n,) event timestamp
ctx.data.y_raw            int8  (n,) the long_view label
ctx.col(name)             any log column, one value PER ROW: 'tab','duration_ms',
                          'hourmin','play_time_ms','is_click','is_like','is_follow',
                          'is_comment','is_forward'
ctx.video_attr(name)      video attribute BROADCAST TO ROWS: 'author_id','music_id',
                          'video_type','upload_type','video_duration','tag','music_type'
ctx.user_attr(name)       user attribute broadcast to rows: 'user_active_degree',
                          'follow_user_num_range','fans_user_num_range','onehot_feat0'...
                          (constant within a user, so useless alone - cross it with
                          something item-varying)
ctx.check(X, names)       validate your matrix before returning; raises the same errors
                          the harness would, so you can fix them in place
NOTE ctx.col(name,'vb') is indexed by video_id (~7.5k entries), NOT by row. Never reshape
it - use ctx.video_attr(name).
ctx.baseline_score        float32 (n,) the official FM baseline's OUT-OF-SAMPLE score for
                          each row (trained per window on data before it). This is the
                          model you are trying to beat - include it as a feature and build
                          on it rather than trying to rediscover it.
ctx.mf_factors(dim=16)    -> (U, V) float32 (n,dim) each. Implicit-ALS collaborative-
                          filtering embeddings, leakage-safe per frozen window. A
                          DIFFERENT inductive bias than the FM's per-ID crosses (a low-rank
                          factorization of the whole 0.58%-dense interaction matrix), so
                          combining it with baseline_score is more promising than either
                          alone. dot(U[i],V[i]) is a CF score; the raw vectors can also be
                          used as per-dimension features.
ctx.auxiliary_signal(name) -> float32 (n,) out-of-sample propensity for another feedback
                          signal. name in {'is_click','is_like','is_follow','is_comment',
                          'is_forward'}. The legitimate route to these signals - their raw
                          columns are blocked at ctx.col() because they are outcomes of
                          THIS row and would leak the answer.
ctx.cf_score()            -> (score, hist_count) float32 (n,) each. IDF-weighted item-item
                          CF: mean similarity between the row's item and this user's
                          frozen-history items. hist_count is a confidence weight (0 = cold
                          start).
ctx.fold.idx['train'|'valid']   row indices    ctx.fold.horizon   last date with labels
ctx.OFFICIAL_WINDOWS            frozen-window schedule
ctx.window_horizons(date, windows) -> per-row horizon
ctx.frozen_prefix(keys, date, y, labeled, horizon_per_row) -> (n_labeled, n_pos) per row,
                          counted over rows sharing `keys` dated <= that row's horizon.
                          EVERY argument is a flat 1-D array of length ctx.data.n. For a
                          composite key, factorize it into ONE 1-D array first:
                          np.unique(np.stack([a,b],1), axis=0, return_inverse=True)[1]
ctx.causal_prefix(keys, time_ms, y, labeled) -> (n_before, n_labeled, n_pos), STREAMING
                          prefix. Read its docstring: it is not a correct model of this
                          task on its own.
ctx.smoothed_rate(pos, labeled, prior, alpha) -> beta-smoothed rate
train_cfg (optional): {'objective': 'binary'|'lambdarank', 'group': 'user_day'|'user',
  'hparams': {...}, 'mode': 'features'|'scores'}
  mode='features' (default): X is a feature matrix. The harness trains a fresh LightGBM on
    it per seed and averages, using the objective/group/hparams above.
  mode='scores': X (shape (n,1)) is the FINAL per-row score, already fully computed by
    YOUR OWN code - e.g. you trained your own lightgbm model(s) inside build() and blended
    them with ctx.baseline_score / ctx.cf_score / etc. yourself (numpy: within-user
    percentile rank, then a weighted average). The harness evaluates X directly, no refit.
    USE THIS TO ENSEMBLE MULTIPLE MODELS. Concatenating every signal into one feature
    matrix for a single downstream tree is a DIFFERENT, WEAKER strategy that has already
    been tried three times live and lost every time (LightGBM shatters a smooth, already-
    good continuous score - like ctx.baseline_score - into step-function splits, which
    degrades it; feeding it back in as a raw column does not fix this). Blending the FINAL
    OUTPUTS of separately-trained models is the strategy that actually won by hand on this
    benchmark (+0.0030 primary). If you train anything stochastic (e.g. lightgbm) inside
    build(), loop over 2-3 seeds YOURSELF and average - the harness calls build() only
    once per candidate, so it cannot average across calls for you in this mode. `lightgbm`
    is import-allowed; do not import `torch` (it will crash if loaded alongside lightgbm
    in the same process).

Allowed imports: numpy, scipy, math, collections, itertools. No file I/O, no network.

Respond with STRICT JSON only:
{"hypothesis": {"statement": "...", "mechanism": "...", "predicted_effect": "which slice
moves and which way", "predicted_gain": 0.004, "family": "history|debias|ensemble|regime|
objective|capacity"}, "code": "def build(ctx):\\n    ..."}"""


class Proposal:
    def __init__(self, hypothesis, code, raw=''):
        self.hypothesis, self.code, self.raw = hypothesis, code, raw


_UESC = re.compile(r'\\u([0-9a-fA-F]{4})')


def _unescape(v):
    """Decode literal \\uXXXX sequences that survive JSON decoding.

    Models sometimes emit a double-escaped sequence, so json.loads yields the six
    characters rather than the character. Harmless for code, but the hypothesis text goes
    straight into the run log that judges read, so normalise it."""
    if isinstance(v, str):
        return _UESC.sub(lambda m: chr(int(m.group(1), 16)), v)
    if isinstance(v, dict):
        return {k: _unescape(x) for k, x in v.items()}
    return v


def _extract_json(text):
    m = re.search(r'\{.*\}', text, re.S)
    if not m:
        raise ValueError('no JSON object in response')
    return json.loads(m.group(0))


PROPOSAL_SCHEMA = {
    "type": "object",
    "properties": {
        "hypothesis": {
            "type": "object",
            "properties": {
                "statement": {"type": "string"},
                "mechanism": {"type": "string"},
                "predicted_effect": {"type": "string"},
                "predicted_gain": {"type": "number"},
                "family": {"type": "string"},
            },
            "required": ["statement", "mechanism", "predicted_effect",
                         "predicted_gain", "family"],
            "additionalProperties": False,
        },
        "code": {"type": "string"},
    },
    "required": ["hypothesis", "code"],
    "additionalProperties": False,
}


class AnthropicProposer:
    """Claude via the Anthropic SDK.

    Two things here are easy to get wrong and cost us a whole run when we did:

    * Thinking is ON BY DEFAULT on Claude Opus 5 / Sonnet 5. With max_tokens=4000 the
      model spent the entire budget inside a `thinking` block and returned zero text
      (stop_reason='max_tokens'). max_tokens must cover thinking AND the answer.
    * Free-text JSON is not guaranteed to parse. `output_config.format` constrains the
      response to our schema, so the proposal either arrives well-formed or the request
      fails loudly - no regex salvage, no silent malformed proposals.

    Effort is set to 'medium' deliberately: the task is bounded and the schema is strict,
    so buying more thinking mostly buys tokens, and token spend is a scored criterion.
    """

    def __init__(self, model='claude-opus-5', max_tokens=16000, api_key=None,
                 workspace_id=None, effort='medium'):
        from anthropic import Anthropic
        ws = workspace_id or os.environ.get('ANTHROPIC_WORKSPACE_ID')
        headers = {'anthropic-workspace-id': ws} if ws else None
        self.client = Anthropic(api_key=api_key or os.environ.get('ANTHROPIC_API_KEY'),
                                default_headers=headers)
        self.model, self.max_tokens, self.effort = model, max_tokens, effort
        self.tokens_in = self.tokens_out = 0

    def propose(self, digest, ledger_summary, budget, last_failure=None):
        user = {'current_diagnostics': digest, 'history': ledger_summary, 'budget': budget}
        if last_failure:
            user['previous_attempt_failed'] = last_failure
        msg = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens, system=SYSTEM,
            output_config={'effort': self.effort,
                           'format': {'type': 'json_schema', 'schema': PROPOSAL_SCHEMA}},
            messages=[{'role': 'user', 'content': json.dumps(user, default=float)}])
        self.tokens_in += msg.usage.input_tokens
        self.tokens_out += msg.usage.output_tokens
        if msg.stop_reason == 'max_tokens':
            raise RuntimeError(
                f"proposal truncated at max_tokens={self.max_tokens}; thinking plus the "
                f"answer did not fit. Raise max_tokens or lower effort.")
        if msg.stop_reason == 'refusal':
            raise RuntimeError(f"model declined: {getattr(msg, 'stop_details', None)}")
        text = ''.join(b.text for b in msg.content if b.type == 'text')
        if not text.strip():
            raise RuntimeError(f"empty text response (stop_reason={msg.stop_reason}, "
                               f"blocks={[b.type for b in msg.content]})")
        d = json.loads(text)
        return Proposal(_unescape(d['hypothesis']), d['code'], text)


class MockProposer:
    """Deterministic scripted proposals - lets the whole loop be tested without an API key,
    and gives a reproducible run for the writeup. Includes a deliberately LEAKING candidate
    so the auditor's catch-and-recover path is exercised on every test run."""

    def __init__(self):
        self.tokens_in = self.tokens_out = 0
        self.i = -1

    SCRIPT = [
        # 1. the natural, wrong first move: streaming-causal history aggregates
        dict(hypothesis=dict(
            statement="User and item history rates should add personalisation the ID model lacks",
            mechanism="FM only sees IDs; explicit long_view rates summarise behaviour directly",
            predicted_effect="GAUC rises across all user-activity deciles",
            predicted_gain=0.01, family='history'),
            code=textwrap.dedent('''
            import numpy as np
            def build(ctx):
                d = ctx.data
                y = d.y_raw.astype(np.float64)
                lab = (d.date <= ctx.fold.horizon)
                cols, names = [], []
                for nm, keys in (('item', d.video_id.astype(np.int64)),
                                 ('user', d.user_id.astype(np.int64))):
                    n_, l_, p_ = ctx.causal_prefix(keys, d.time_ms, y, lab)
                    cols.append(ctx.smoothed_rate(p_, l_, 0.33, 20.0)); names.append(nm+'_rate')
                    cols.append(np.log1p(l_)); names.append(nm+'_logn')
                return np.stack(cols, 1).astype(np.float32), names
            ''').strip()),
        # 2. after the auditor rejects it: the frozen construction
        dict(hypothesis=dict(
            statement="Freeze history at each evaluation window's start to remove list-mate label feedback",
            mechanism="Evaluation ranks a user's list as a set, so labels from inside that "
                      "list do not exist at scoring time and must not enter features",
            predicted_effect="validation falls toward test; the val-test gap collapses",
            predicted_gain=0.0, family='debias'),
            code=textwrap.dedent('''
            import numpy as np
            def build(ctx):
                d = ctx.data
                y = d.y_raw.astype(np.float64)
                hz = ctx.window_horizons(d.date.astype(np.int64), ctx.OFFICIAL_WINDOWS)
                lab = np.ones(d.n, dtype=bool)
                cols, names = [], []
                for nm, keys in (('item', d.video_id.astype(np.int64)),
                                 ('user', d.user_id.astype(np.int64))):
                    l_, p_ = ctx.frozen_prefix(keys, d.date.astype(np.int64), y, lab, hz)
                    cols.append(ctx.smoothed_rate(p_, l_, 0.33, 20.0)); names.append(nm+'_rate')
                    cols.append(np.log1p(l_)); names.append(nm+'_logn')
                return np.stack(cols, 1).astype(np.float32), names
            ''').strip()),
    ]

    def propose(self, digest, ledger_summary, budget, last_failure=None):
        self.i = min(self.i + 1, len(self.SCRIPT) - 1)
        s = self.SCRIPT[self.i]
        self.tokens_in += 1200; self.tokens_out += 400
        return Proposal(s['hypothesis'], s['code'], '<mock>')


class OpenAICompatibleProposer:
    """Any OpenAI-shaped /chat/completions endpoint.

    Covers Volcengine Ark (Doubao), OpenRouter, DeepSeek, OpenAI, Azure, and a local
    Ollama server - which is also how Trae Agent talks to most providers, so a key that
    works in Trae works here.  Kept deliberately dependency-light (httpx ships with the
    anthropic package) so no extra install is needed.

    base_url examples
        Volcengine Ark : https://ark.cn-beijing.volces.com/api/v3
        OpenRouter     : https://openrouter.ai/api/v1
        DeepSeek       : https://api.deepseek.com/v1
        Ollama (local) : http://localhost:11434/v1        (api_key can be anything)
    """

    def __init__(self, base_url, model, api_key=None, max_tokens=4000, timeout=180,
                 temperature=0.6):
        import httpx
        self.url = base_url.rstrip('/') + '/chat/completions'
        self.model, self.max_tokens, self.temperature = model, max_tokens, temperature
        self.key = api_key or os.environ.get('LLM_API_KEY', 'none')
        self.client = httpx.Client(timeout=timeout)
        self.tokens_in = self.tokens_out = 0

    def propose(self, digest, ledger_summary, budget, last_failure=None):
        user = {'current_diagnostics': digest, 'history': ledger_summary, 'budget': budget}
        if last_failure:
            user['previous_attempt_failed'] = last_failure
        body = {'model': self.model, 'max_tokens': self.max_tokens,
                'temperature': self.temperature,
                'messages': [{'role': 'system', 'content': SYSTEM},
                             {'role': 'user', 'content': json.dumps(user, default=float)}]}
        r = self.client.post(self.url, json=body,
                             headers={'Authorization': f'Bearer {self.key}',
                                      'Content-Type': 'application/json'})
        r.raise_for_status()
        j = r.json()
        u = j.get('usage') or {}
        self.tokens_in += u.get('prompt_tokens', 0)
        self.tokens_out += u.get('completion_tokens', 0)
        text = j['choices'][0]['message']['content']
        d = _extract_json(text)
        return Proposal(d['hypothesis'], d['code'], text)


def make_proposer(spec):
    """spec: 'mock' | 'pool' | 'anthropic:<model>' | '<base_url>|<model>'"""
    if spec == 'mock':
        return MockProposer()
    if spec == 'pool':
        from kairos.agent.proposer_pool import PoolProposer
        return PoolProposer()
    if spec == 'two-stage':
        return TwoStageProposer()
    if spec.startswith('two-stage:'):
        planner, coder = spec.split(':', 1)[1].split('+')
        return TwoStageProposer(planner=planner, coder=coder)
    if spec.startswith('anthropic:'):
        return AnthropicProposer(model=spec.split(':', 1)[1])
    base, model = spec.split('|', 1)
    return OpenAICompatibleProposer(base, model)


PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "statement": {"type": "string"},
        "mechanism": {"type": "string"},
        "predicted_effect": {"type": "string"},
        "predicted_gain": {"type": "number"},
        "family": {"type": "string"},
        "implementation_sketch": {"type": "string"},
    },
    "required": ["statement", "mechanism", "predicted_effect", "predicted_gain",
                 "family", "implementation_sketch"],
    "additionalProperties": False,
}
CODE_SCHEMA = {"type": "object", "properties": {"code": {"type": "string"}},
               "required": ["code"], "additionalProperties": False}

PLANNER_SYSTEM = SYSTEM.split('YOU WRITE PYTHON.')[0] + """
YOU DO NOT WRITE CODE. Decide WHAT to try next and WHY, grounded in the diagnostics you
are given rather than in general recommendations.

BUDGET IS A HARD CONSTRAINT, NOT A HINT. The run ENDS after 3 consecutive iterations
without a +0.002 validation gain. `budget.misses_before_run_ends` tells you how many
failures you can still absorb. With 1 remaining, propose a change with a high probability
of a small gain; with 3, an exploratory swing is affordable. Say which you are doing. Commit to a falsifiable prediction: name
the diagnostic slice that should move and the direction. Then describe the implementation
as a short sketch for an engineer - which keys, which aggregates, which columns.

Respond with STRICT JSON only:
{"statement": "...", "mechanism": "...", "predicted_effect": "...",
 "predicted_gain": 0.004, "family": "history|debias|ensemble|regime|objective|capacity",
 "implementation_sketch": "..."}"""

CODER_SYSTEM = ("You implement ONE feature-construction function exactly as specified.\n"
                "Do not redesign the idea; implement the sketch you are given.\n\n"
                + SYSTEM[SYSTEM.index('YOU WRITE PYTHON.'):].split('Respond with STRICT')[0]
                + '\nRespond with STRICT JSON only: {"code": "def build(ctx):\\n    ..."}')


class TwoStageProposer:
    """Opus plans, Sonnet implements - and repairs never re-plan.

    Splitting the roles matches how the work actually divides. Deciding what to try next
    is judgement over evidence; turning an agreed sketch into numpy is mechanical. Running
    both on the strongest model pays a premium for the half that does not need it.

    The bigger win is the repair path. When a candidate dies on a traceback the HYPOTHESIS
    is still sound - only the code is wrong - so `repair()` re-invokes the coder alone,
    carrying the same hypothesis forward. Re-planning on every failure re-derives reasoning
    that was never in question, and token spend is a scored criterion here.
    """

    def __init__(self, planner='claude-opus-5', coder='claude-sonnet-5',
                 api_key=None, workspace_id=None, plan_effort='high',
                 code_effort='low', max_tokens=16000):
        from anthropic import Anthropic
        ws = workspace_id or os.environ.get('ANTHROPIC_WORKSPACE_ID')
        headers = {'anthropic-workspace-id': ws} if ws else None
        self.client = Anthropic(api_key=api_key or os.environ.get('ANTHROPIC_API_KEY'),
                                default_headers=headers)
        self.planner, self.coder = planner, coder
        self.plan_effort, self.code_effort = plan_effort, code_effort
        self.max_tokens = max_tokens
        self.tokens_in = self.tokens_out = 0
        self.by_model = {}
        self.last_plan = None

    def _call(self, model, system, payload, schema, effort):
        msg = self.client.messages.create(
            model=model, max_tokens=self.max_tokens, system=system,
            output_config={'effort': effort,
                           'format': {'type': 'json_schema', 'schema': schema}},
            messages=[{'role': 'user', 'content': json.dumps(payload, default=float)}])
        self.tokens_in += msg.usage.input_tokens
        self.tokens_out += msg.usage.output_tokens
        b = self.by_model.setdefault(model, {'in': 0, 'out': 0, 'calls': 0})
        b['in'] += msg.usage.input_tokens; b['out'] += msg.usage.output_tokens; b['calls'] += 1
        if msg.stop_reason == 'max_tokens':
            raise RuntimeError(f"{model}: truncated at max_tokens={self.max_tokens}")
        text = ''.join(x.text for x in msg.content if x.type == 'text')
        if not text.strip():
            raise RuntimeError(f"{model}: empty response (stop_reason={msg.stop_reason})")
        return json.loads(text)

    def propose(self, digest, ledger_summary, budget, last_failure=None):
        plan = self._call(self.planner, PLANNER_SYSTEM,
                          {'current_diagnostics': digest, 'history': ledger_summary,
                           'budget': budget}, PLAN_SCHEMA, self.plan_effort)
        self.last_plan = plan
        return self._implement(plan, last_failure)

    def repair(self, hypothesis, failure):
        """Same hypothesis, new code. The planner is deliberately not consulted."""
        return self._implement(self.last_plan or hypothesis, failure)

    def _implement(self, plan, failure=None):
        payload = {'implement_this': plan}
        if failure:
            payload['previous_attempt_failed'] = failure
            payload['instruction'] = ('Fix the error. Keep the same idea; change only what '
                                      'the traceback requires.')
        d = self._call(self.coder, CODER_SYSTEM, payload, CODE_SCHEMA, self.code_effort)
        hyp = _unescape({k: plan[k] for k in ('statement', 'mechanism',
                                              'predicted_effect', 'predicted_gain',
                                              'family')})
        return Proposal(hyp, d['code'], json.dumps(plan))
