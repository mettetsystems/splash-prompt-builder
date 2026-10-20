"""Project-local processing checkpoints and explicit wording decisions."""
import hashlib
import json
import os
from difflib import SequenceMatcher
import psutil

from backend.storage import now, uid

LANGUAGE = 'Language and instruction precision'


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def initialize(project):
    legacy = 'edit_decisions' not in project
    project.setdefault('processing', {'runs': {}, 'settled': None})
    project.setdefault('edit_decisions', [])
    if legacy:
        text = project['draft']['text']
        for c in project.get('comments', []):
            if c['kind'] != 'suggestion' or c['status'] not in ('accepted','resolved','dismissed'):
                continue
            edit = comment_edit(c)
            if c['status']=='accepted' and c.get('replacement') and text.count(c['replacement'])==1:
                remember(project, edit, 'accepted', start=text.index(c['replacement']), quote=c['replacement'], comment_id=c['id'])
            elif not c['outdated'] and text[edit['start']:edit['end']]==edit['before']:
                remember(project, edit, c['status'], comment_id=c['id'])
        project['proposals'] = [e for e in project.get('proposals', []) if allowed(project, e)]
        metadata = project['draft'].get('metadata', {})
        if metadata.get('accepted_edits') or metadata.get('variant_id'):
            settle(project, 'prior_author_choice')
        elif sum(r['kind']=='main' for r in project['revisions'])>1 and any(r['id']==project['main_id'] and r['text']==text for r in project['revisions']):
            settle(project, 'prior_confirmed_main')


def input_key(project, kind, model, focus=None):
    # Generated suggestions, timestamps, latency, and query results are outputs,
    # never inputs to this key: they must not cause another automatic cycle.
    inputs = [kind, project['agent'], project['draft']['text'],
              [model.get('model'), model.get('config', {})] if model.get('state') == 'ready' else None]
    if kind == 'refine':
        from backend.enrichment import PIPELINE_VERSION
        inputs += [PIPELINE_VERSION, project['settings']['output'], project['documents'], focus,
                   next(r['text'] for r in project['revisions'] if r['id'] == project['main_id']),
                   [[c['fingerprint'], c['status'], c['outdated']] for c in project['comments'] if c['kind'] == 'conflict']]
    if kind == 'research':
        inputs += [project['settings'].get('scholar'), project['settings'].get('research')]
    return digest(inputs)


def settle(project, reason):
    initialize(project)
    project['processing']['settled'] = {'text_hash': digest(project['draft']['text']),
        'version': project['draft']['version'], 'reason': reason, 'created': now()}


def is_settled(project):
    saved = project.get('processing', {}).get('settled')
    return bool(saved and saved['text_hash'] == digest(project['draft']['text']))


def interrupted(run):
    if run['status'] != 'running':
        return False
    if run.get('owner_pid', 0) <= 0:
        return True
    try:
        os.kill(run['owner_pid'], 0)
        process = psutil.Process(run['owner_pid'])
        return process.status()==psutil.STATUS_ZOMBIE or (run.get('owner_started') is not None and process.create_time()!=run['owner_started'])
    except (ProcessLookupError, psutil.NoSuchProcess):
        return True
    except (PermissionError, psutil.AccessDenied):
        pass  # A live process owned by another user is still live.
    return False


def claim(project, kind, key, force=False):
    initialize(project)
    runs = project['processing']['runs'].setdefault(kind, [])
    for run in runs:
        if interrupted(run):
            run.update(status='canceled', finished=now(), error='The previous server stopped. Retry manually when ready.')
    running = next((r for r in reversed(runs) if r['status'] == 'running'), None)
    if running:
        return None, {'status':'already_running'}
    if not force and is_settled(project):
        return None, {'status':'unchanged', 'reason':'Your chosen draft is saved. Waiting for new input.', 'result':{'settled':True}}
    prior = next((r for r in reversed(runs) if r['key'] == key), None)
    if prior and not force:
        reason = 'Already reviewed this content. Waiting for new input.' if prior['status'] == 'complete' else 'Previous attempt stopped. Edit the draft or retry manually.'
        return None, {'status':'unchanged', 'reason':reason, 'result':{**prior.get('result', {}), 'cached':True}, 'outcome':prior['status']}
    run = {'id':uid(), 'key':key, 'status':'running', 'version':project['draft']['version'],
           'owner_pid':os.getpid(), 'owner_started':psutil.Process().create_time(), 'created':now()}
    runs.append(run)
    # Retain the current run and recent input checkpoints, without storing copies
    # of entire prompts or model output in this ledger.
    project['processing']['runs'][kind] = runs[-128:]
    return run['id'], None


def finish(project, kind, run_id, status, result=None, error=None):
    run = next((r for r in project['processing']['runs'].get(kind, []) if r['id'] == run_id), None)
    if not run:
        return
    summary = {k:v for k,v in (result or {}).items() if k in ('comments_added','review_required','conflicts','errors','context','suppressed_edits','quality')}
    if 'proposals' in (result or {}):
        summary['proposals_count'] = len(result['proposals'])
    run.update(status=status, result=summary, finished=now())
    if error:
        run['error'] = error


def remember(project, edit, status, *, start=None, quote=None, comment_id=None):
    initialize(project)
    start = edit['start'] if start is None else start
    quote = edit['before'] if quote is None else quote
    decision = {'id':uid(), 'status':status, 'before':edit['before'], 'after':edit['after'],
        'start':start, 'end':start+len(quote), 'quote':quote, 'active':True,
        'category':'language' if edit.get('rationale', LANGUAGE)==LANGUAGE else 'research',
        'comment_id':comment_id, 'sources':edit.get('sources', []), 'created':now()}
    project['edit_decisions'].append(decision)
    return decision


def comment_edit(comment):
    a = comment['anchors'][0]
    return {'start':a['start'], 'end':a['end'], 'before':a['quote'], 'after':comment['replacement'], 'sources':comment['sources']}


def overlaps(a, b):
    if a['start'] == a['end'] or b['start'] == b['end']:
        return a['start'] <= b['end'] and b['start'] <= a['end']
    return max(a['start'], b['start']) < min(a['end'], b['end'])


def allowed(project, edit):
    for decision in project.get('edit_decisions', []):
        if not decision['active']:
            continue
        if edit.get('rationale', LANGUAGE) == LANGUAGE:
            if (decision.get('category','language')=='language' or decision['status']=='accepted') and overlaps(decision, edit):
                return False
        elif set(decision.get('sources', [])) & set(edit.get('sources', [])):
            return False
    return True


def reanchor_decisions(project, old, new):
    matches = SequenceMatcher(None, old, new, autojunk=len(old)>12000).get_matching_blocks()
    for d in project.get('edit_decisions', []):
        if not d['active']:
            continue
        block = next((b for b in matches if b.size and b.a<=d['start'] and d['end']<=b.a+b.size), None)
        if block is None:
            d['active'] = False  # The author edited this passage: it may be reviewed anew.
        else:
            delta = block.b-block.a
            d['start'] += delta
            d['end'] += delta


def reject_edits(project, edits, status='dismissed', comment_id=None):
    for edit in edits:
        remember(project, edit, status, comment_id=comment_id)
    # This also removes edits created by older versions, whose character-level
    # spans sit inside the word-level comment the user dismissed.
    project['proposals'] = [e for e in project['proposals'] if allowed(project, e)]
    for c in project['comments']:
        if c['kind']=='suggestion' and c['status']=='open' and not c['outdated'] and not allowed(project, comment_edit(c)):
            c.update(status=status, updated=now())


def accept_edits(project, edits):
    # Called after replacing the draft; convert original offsets to the resulting
    # text, including all prior selected edits. Protect approved wording against
    # automatic reversal until the author changes the passage.
    for edit in edits:
        shift = sum(len(e['after'])-(e['end']-e['start']) for e in edits if e is not edit and e['end']<=edit['start'])
        start = edit['start']+shift
        remember(project, edit, 'accepted', start=start, quote=edit['after'])
    settle(project, 'accepted_edits')


def mark_accepted_threads(project, edits):
    for comment in project.get('comments', []):
        if comment['kind']!='suggestion' or comment['status']!='open' or comment['outdated']:
            continue
        a = comment['anchors'][0]
        contained = [e for e in edits if a['start']<=e['start'] and e['end']<=a['end']]
        result = a['quote']
        for edit in sorted(contained, key=lambda e:e['start'], reverse=True):
            start,end = edit['start']-a['start'],edit['end']-a['start']
            result = result[:start]+edit['after']+result[end:]
        if contained and result==comment['replacement']:
            comment.update(status='accepted', updated=now())


def decision_context(project):
    decided = [c for c in project.get('comments', []) if c['status'] in ('resolved','dismissed','accepted')]
    lines = [f"{c['status']}: {next((m['text'] for m in reversed(c['messages']) if m['author']=='you'), c['messages'][0]['text'])[:140]}" for c in decided[-4:]]
    protected = [d for d in project.get('edit_decisions', []) if d['active']]
    lines += [f"Keep the author's chosen wording: {d['quote'][:120]}" for d in protected[-4:]]
    return ('\nPrior author decisions (do not repeat resolved questions or undo chosen wording):\n'+'\n'.join(lines)) if lines else ''
