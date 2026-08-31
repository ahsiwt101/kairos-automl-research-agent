"""Execute LLM-authored code safely, and turn every failure into a usable message.

Robustness on this track is explicitly scored on how a failure is handled, not on how
often one happens, so the contract here is: nothing the model writes can corrupt the run,
and every rejection produces a diagnosis specific enough for the next attempt to fix it.

Three layers:
  static   an AST pass rejects imports outside an allowlist, filesystem writes, network,
           exec/eval, and dunder access before anything runs.
  process  the candidate runs in a subprocess with a wall-clock timeout, so an infinite
           loop or a segfault costs one iteration rather than the run.
  semantic the temporal-validity auditor (kairos.agent.auditor) runs on the OUTPUT, which
           is what catches the failure mode that actually matters here - code that runs
           perfectly and produces a leaking feature.
"""
import ast, json, os, subprocess, sys, tempfile, textwrap, time

ALLOWED_IMPORTS = {
    'numpy', 'np', 'math', 'json', 'itertools', 'collections', 'functools',
    'scipy', 'scipy.sparse', 'scipy.stats', 'lightgbm', 'sklearn',
    'kairos', 'kairos.kernel', 'kairos.kernel.causal', 'kairos.kernel.frozenfeat',
    'kairos.kernel.fastmetrics', 'kairos.kernel.dataset', 'kairos.kernel.diagnostics',
}
FORBIDDEN_CALLS = {'eval', 'exec', 'compile', '__import__', 'open', 'input',
                   'globals', 'locals', 'getattr', 'setattr', 'delattr'}


class StaticViolation(Exception):
    pass


def static_check(src):
    """Reject dangerous constructs before execution. Returns list of findings."""
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        raise StaticViolation(f"syntax error line {e.lineno}: {e.msg}")
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                root = a.name.split('.')[0]
                if root not in ALLOWED_IMPORTS and a.name not in ALLOWED_IMPORTS:
                    bad.append(f"import '{a.name}' not in the allowlist")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ''
            if mod.split('.')[0] not in ALLOWED_IMPORTS and mod not in ALLOWED_IMPORTS:
                bad.append(f"from '{mod}' import ... not in the allowlist")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALLS:
                bad.append(f"call to '{node.func.id}()' is not permitted")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith('__') and node.attr.endswith('__'):
                bad.append(f"dunder attribute access '{node.attr}'")
    if bad:
        raise StaticViolation('; '.join(sorted(set(bad))))
    return True


CANDIDATE_CONTRACT = '''
# A candidate module must define exactly this function:
#
#   def build(ctx):        # may return (X, names) or (X, names, train_cfg)
#       # train_cfg keys: objective ('binary'|'lambdarank'), group ('user_day'|'user'),
#       #   hparams ({...}), mode ('features'|'scores'). mode='scores' means X (n,1) is
#       #   the FINAL score you already computed (e.g. by training your own model(s) and
#       #   blending them) - see proposer.py's SYSTEM prompt for the full rationale.
#       """ctx exposes:
#            ctx.data            columnar log (user_id, video_id, date, time_ms, tab, ...)
#            ctx.fold            train/valid indices; TEST LABELS ARE NOT REACHABLE
#            ctx.causal_prefix   streaming prefix aggregates (see the docstring: this is
#                                NOT a correct model of this task on its own)
#            ctx.frozen_prefix   window-frozen aggregates
#            ctx.window_horizons map rows to their window's frozen horizon
#          returns (X float32 (N, F), names list[str])
#       """
'''


RUNNER = r'''
import sys, json, time, traceback
sys.path.insert(0, {root!r})
import numpy as np
def main():
    import importlib.util
    spec = importlib.util.spec_from_file_location('candidate', {mod!r})
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    from kairos.agent.context import make_context
    ctx = make_context({fold!r})
    t0 = time.time()
    built = m.build(ctx)
    cfg = {{}}
    if len(built) == 3:
        X, names, cfg = built
    else:
        X, names = built
    # Validate train_cfg KEYS here, in the same subprocess and same repair loop as the
    # candidate's own code - not in the later, separate scoring subprocess, which has no
    # repair path and would silently discard a build() that ran perfectly correctly.
    from kairos.agent.traincfg import validate_keys as _validate_hparam_keys
    _validate_hparam_keys((cfg or {{}}).get('hparams'))
    X = np.asarray(X, dtype=np.float32)
    assert X.ndim == 2, f"build() must return a 2-D matrix, got shape {{X.shape}}"
    assert X.shape[0] == ctx.data.n, (
        f"build() returned {{X.shape[0]}} rows but the log has {{ctx.data.n}}; "
        f"features must be aligned to ALL rows in data order")
    assert len(names) == X.shape[1], (
        f"{{len(names)}} names for {{X.shape[1]}} columns")
    assert np.isfinite(X).all(), "feature matrix contains NaN or Inf"
    np.save({out!r}, X)
    json.dump({{'names': list(names), 'seconds': round(time.time()-t0,1),
               'train_cfg': dict(cfg or {{}})}}, open({meta!r}, 'w'))
    print("OK")
try:
    main()
except Exception:
    traceback.print_exc(); sys.exit(1)
'''


def _run_capped(cmd, timeout, mem_limit_gb, poll=0.5):
    """Run a candidate under BOTH a time budget and a memory budget.

    Why not resource.setrlimit(RLIMIT_AS): on macOS that call raises OSError outright, so
    a preexec_fn that sets it either kills every launch or - if it swallows the error, as
    a first version here did - silently caps nothing. The guard looked present and did
    nothing, which is worse than having none.

    Polling the child's RSS from the parent works on any platform and measures resident
    pages, which is what actually triggers the OS killer. Without it, a candidate that
    over-allocates gets the PARENT killed (it holds the columnar cache and every prewarmed
    signal, so it is the largest process) and the whole run vanishes with no traceback and
    no ledger entry. With it, the child dies first and the failure comes back through the
    normal path as something the agent can read and repair around.
    """
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    cap_mb = int(mem_limit_gb * 1024) if mem_limit_gb and mem_limit_gb > 0 else 0
    t0, killed = time.time(), None
    while proc.poll() is None:
        elapsed = time.time() - t0
        if elapsed > timeout:
            killed = 'timeout'
            break
        if cap_mb:
            try:
                out = subprocess.run(['ps', '-o', 'rss=', '-p', str(proc.pid)],
                                     capture_output=True, text=True, timeout=5).stdout.strip()
                if out and int(out) / 1024.0 > cap_mb:
                    killed = 'memory'
                    break
            except (ValueError, OSError, subprocess.SubprocessError):
                pass          # cannot read RSS; the time budget still bounds the run
        time.sleep(poll)
    if killed:
        proc.kill()
        proc.communicate()
        return None, killed
    out, err = proc.communicate()
    return subprocess.CompletedProcess(cmd, proc.returncode, out, err), None


def run_candidate(src, fold_name='official', timeout=900, workdir='runs/sandbox',
                  mem_limit_gb=float(os.environ.get('KAIROS_CAND_MEM_GB', '6'))):
    """Static-check, then execute in a subprocess. Returns a result dict.

    Never raises on candidate failure - a failure is data the agent needs, so it comes
    back as {'ok': False, 'stage': ..., 'error': ...} with the traceback trimmed to
    something a language model can act on.
    """
    os.makedirs(workdir, exist_ok=True)
    try:
        static_check(src)
    except StaticViolation as e:
        return {'ok': False, 'stage': 'static', 'error': str(e),
                'hint': 'rewrite using only the allowlisted imports and no I/O'}

    mod = os.path.join(workdir, 'candidate.py')
    out = os.path.join(workdir, 'X.npy')
    meta = os.path.join(workdir, 'meta.json')
    with open(mod, 'w') as fh:
        fh.write(src)
    root = os.path.abspath('.')
    runner = os.path.join(workdir, 'runner.py')
    with open(runner, 'w') as fh:
        fh.write(RUNNER.format(root=root, mod=os.path.abspath(mod), fold=fold_name,
                               out=os.path.abspath(out), meta=os.path.abspath(meta)))
    t0 = time.time()
    try:
        p, killed = _run_capped([sys.executable, runner], timeout, mem_limit_gb)
    except subprocess.TimeoutExpired:
        killed, p = 'timeout', None
    if killed == 'timeout':
        return {'ok': False, 'stage': 'timeout', 'seconds': timeout,
                'error': f'candidate exceeded the {timeout}s budget',
                'hint': 'the construction is too slow; vectorise it or reduce its scope'}
    if killed == 'memory':
        return {'ok': False, 'stage': 'memory', 'seconds': round(time.time() - t0, 1),
                'error': (f'candidate exceeded the {mem_limit_gb:g} GB memory budget and '
                          f'was terminated'),
                'hint': ('materialise fewer full-length columns: build features in float32 '
                         'rather than float64, delete intermediates you no longer need, '
                         'and avoid np.unique over a full-length 2-column stack when a '
                         'factorised key would do')}
    if p.returncode != 0:
        tb = p.stderr.strip().splitlines()
        return {'ok': False, 'stage': 'runtime', 'seconds': round(time.time()-t0, 1),
                'error': '\n'.join(tb[-12:]),
                'hint': 'fix the exception; the contract is in CANDIDATE_CONTRACT'}
    info = json.load(open(meta))
    return {'ok': True, 'X_path': out, 'names': info['names'],
            'train_cfg': info.get('train_cfg', {}),
            'seconds': round(time.time() - t0, 1)}
