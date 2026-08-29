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

YOU WRITE PYTHON. Define exactly one function `build(ctx)` returning (X, names) where X is
a float32 matrix aligned to ALL log rows in data order. Available on ctx: data, fold,
causal_prefix, frozen_prefix, window_horizons, smoothed_rate, OFFICIAL_WINDOWS,
within_user_deviation, col(). Allowed imports: numpy, scipy, math, collections, itertools.
No file I/O, no network.

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


class AnthropicProposer:
    def __init__(self, model='claude-sonnet-5', max_tokens=4000, api_key=None,
                 workspace_id=None):
        from anthropic import Anthropic
        # identity-linked keys must name the workspace the request acts in
        ws = workspace_id or os.environ.get('ANTHROPIC_WORKSPACE_ID')
        headers = {'anthropic-workspace-id': ws} if ws else None
        self.client = Anthropic(api_key=api_key or os.environ.get('ANTHROPIC_API_KEY'),
                                default_headers=headers)
        self.model, self.max_tokens = model, max_tokens
        self.tokens_in = self.tokens_out = 0

    def propose(self, digest, ledger_summary, budget, last_failure=None):
        user = {'current_diagnostics': digest, 'history': ledger_summary, 'budget': budget}
        if last_failure:
            user['previous_attempt_failed'] = last_failure
        msg = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens, system=SYSTEM,
            messages=[{'role': 'user', 'content': json.dumps(user, default=float)}])
        self.tokens_in += msg.usage.input_tokens
        self.tokens_out += msg.usage.output_tokens
        text = ''.join(b.text for b in msg.content if b.type == 'text')
        d = _extract_json(text)
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
