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

YOU WRITE PYTHON. Define exactly one function `build(ctx)` returning (X, names) or
(X, names, train_cfg). X must be float32 with EXACTLY ctx.data.n rows, aligned to all log
rows in data order. Write direct code - the API below is exact, so do not add defensive
fallbacks or try/except probing.

ctx.data.n                int, number of log rows
ctx.data.user_id          int32 (n,)        ctx.data.video_id   int32 (n,)
ctx.data.date             int32 (n,) yyyymmdd
ctx.data.time_ms          int64 (n,) event timestamp
ctx.data.y_raw            int8  (n,) the long_view label
ctx.col(name)             any log column: 'tab','duration_ms','hourmin','play_time_ms',
                          'is_click','is_like','is_follow','is_comment','is_forward'
ctx.col(name,'vb')        video table: 'author_id','music_id','video_type','upload_type'
ctx.col(name,'uf')        user table:  'user_active_degree','follow_user_num_range', ...
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
train_cfg (optional): {'objective': 'binary'|'lambdarank', 'group': 'user_day'|'user'}

Allowed imports: numpy, scipy, math, collections, itertools. No file I/O, no network.

Respond with STRICT JSON only:
{"hypothesis": {"statement": "...", "mechanism": "...", "predicted_effect": "which slice
moves and which way", "predicted_gain": 0.004, "family": "history|debias|ensemble|regime|
objective|capacity"}, "code": "def build(ctx):\\n    ..."}"""


class Proposal:
    def __init__(self, hypothesis, code, raw=''):
        self.hypothesis, self.code, self.raw = hypothesis, code, raw


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
        return Proposal(d['hypothesis'], d['code'], text)


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
    if spec.startswith('anthropic:'):
        return AnthropicProposer(model=spec.split(':', 1)[1])
    base, model = spec.split('|', 1)
    return OpenAICompatibleProposer(base, model)
