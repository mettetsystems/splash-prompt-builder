import React, { useEffect, useRef, useState } from 'react'
import { api, DraftBuffer, safeUrl, isEnrichment, liveEnrichment, codePointSelection, inputSelection, jobAcknowledgement } from './client'
import splashLogo from './assets/splash-logo.jpg'
import { ModelDialog, SystemPromptDialog, NewProjectDialog, Dialog, Diff, SourceLink, AGENTS, stamp, DialogError, RevisionDetails } from './components/StudioParts'
import {CommentRail, CommentedPrompt, EnrichmentReport} from './components/Comments'
import {ContentSampler} from './sampling'

export default function App() {
  const [projects, setProjects] = useState([]), [project, setProject] = useState(null)
  const [editor, setEditor] = useState({ text: '', dirty: false, saving: false })
  const [query, setQuery] = useState(''), [view, setView] = useState('suggestions')
  const [dialog, setDialog] = useState(null), [error, setError] = useState(''), [notice, setNotice] = useState('')
  const [runtime, setRuntime] = useState(null), [jobs, setJobs] = useState({}), [busy, setBusy] = useState(false)
  const [connection, setConnection] = useState('connecting'), [automatic, setAutomatic] = useState(true)
  const [contextInfo, setContextInfo] = useState(null), [allHistory, setAllHistory] = useState(false)
  const [notes, setNotes] = useState(''), [noteName, setNoteName] = useState('Research notes.txt')
  const [showSavedMain, setShowSavedMain] = useState(false)
  const [commenting,setCommenting] = useState(true), [selectedComment,setSelectedComment] = useState(null), [selection,setSelection] = useState(null)
  const buffer = useRef(null), active = useRef(null), textarea = useRef(null), composing = useRef(false), upload = useRef(null)
  const sampler = useRef(null)
  const refreshList = async () => setProjects(await api('/projects'))
  const refreshRuntime = async () => setRuntime(await api('/models'))
  async function guard(operation) {
    setError(''); setBusy(true)
    try { return await operation() } catch (err) { setError(err.message) } finally { setBusy(false) }
  }
  async function startJob(kind,path,body) {
    const id=active.current
    setJobs(previous=>({...previous,[kind]:{status:'running',version:body.expected_version}}))
    try {
      const response=await api(`/projects/${id}${path}`,{method:'POST',body})
      const finished=jobAcknowledgement(response,body.expected_version)
      if(active.current===id&&finished)setJobs(previous=>({...previous,[kind]:finished}))
      return response
    } catch(err) {
      if(active.current===id)setJobs(previous=>({...previous,[kind]:{status:err.status===409?'canceled':'error',error:err.message,version:body.expected_version}}))
      throw err
    }
  }
  async function openProject(id) {
    await buffer.current?.flush()
    const p = await api(`/projects/${id}`)
    buffer.current?.close(); active.current = id; sampler.current = new ContentSampler(p.draft.text); composing.current = false
    setProject(p); setJobs({}); setContextInfo(null); setView(p.agent === 'splash' ? 'context' : 'suggestions'); setAllHistory(false); setNotice(''); setShowSavedMain(false)
    setSelectedComment(null); setSelection(null)
    localStorage.setItem('splash.active', id)
    const key = `splash.recovery.${id}`
    const controller = new DraftBuffer(p.draft, async (text, version) => {
      const saved = await api(`/projects/${id}/draft`, { method: 'PUT', body: { text, expected_version: version } })
      if (active.current === id) setProject(saved)
      return saved.draft
    }, snapshot => { if (active.current === id) setEditor(snapshot) }, value => {
      try { value ? localStorage.setItem(key, JSON.stringify(value)) : localStorage.removeItem(key) } catch { /* server autosave still works */ }
    })
    buffer.current = controller; controller.emit()
    try {
      const recovered = JSON.parse(localStorage.getItem(key) || 'null')
      if (recovered && recovered.text !== p.draft.text) {
        controller.text = recovered.text; controller.version = recovered.version
        if (recovered.version !== p.draft.version) controller.error = Object.assign(new Error('Recovered local text differs from the saved revision. Choose which draft to keep.'), { status: 409 })
        controller.emit(); setNotice('Recovered unsaved text from this browser. Your saved main is unchanged.')
      }
    } catch { /* ignore invalid obsolete browser recovery data */ }
  }
  async function refreshProject(id) {
    const p = await api(`/projects/${id}`)
    if (active.current !== id) return
    if (p.draft.version < (buffer.current?.version || 0)) return
    setProject(p)
    if (!buffer.current?.snapshot().dirty && !buffer.current?.pending) buffer.current?.replace(p.draft)
  }
  async function updateProject(p, discardLocal = false) {
    if (p.id !== active.current) return
    setProject(p)
    if (discardLocal || (!buffer.current.snapshot().dirty && !buffer.current.pending)) buffer.current.replace(p.draft)
    else if (!buffer.current.pending && buffer.current.version !== p.draft.version) {
      buffer.current.error = Object.assign(new Error('The saved draft changed while you were typing. Your local text is preserved.'), {status:409})
      buffer.current.emit()
    }
    await refreshList()
  }
  useEffect(() => {
    let stopped = false
    guard(async () => {
      await Promise.all([refreshList(), refreshRuntime()])
      const id = localStorage.getItem('splash.active')
      if (id && !stopped) try { await openProject(id) } catch { localStorage.removeItem('splash.active') }
    })
    const timer = setInterval(() => refreshRuntime().catch(() => {}), 4000)
    const leaving = event => { if (buffer.current?.snapshot().dirty) { event.preventDefault(); event.returnValue = '' } }
    window.addEventListener('beforeunload', leaving)
    return () => { stopped = true; clearInterval(timer); window.removeEventListener('beforeunload', leaving) }
  }, [])
  useEffect(() => {
    if (!project?.id) return
    const id = project.id
    let stopped = false, socket, reconnect, attempts = 0
    function connect() {
      socket = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws/projects/${id}`)
      socket.onopen = () => { attempts = 0; setConnection('connected'); refreshProject(id).catch(() => {}) }
      socket.onmessage = event => {
        const data = JSON.parse(event.data)
        if (data.project_id !== active.current) return
        if (data.type === 'job_error') setError(data.message)
        if (data.type === 'job_complete') {
          if (data.result?.context) setContextInfo(data.result.context)
          if (data.result?.errors?.length) setNotice(data.result.errors.join(' · '))
          refreshProject(id).catch(() => {})
        }
        if (data.type === 'project_changed' && !buffer.current?.pending) refreshProject(id).catch(() => {})
        if (data.type === 'project_deleted') { active.current = null; buffer.current?.close(); buffer.current = null; setProject(null); refreshList() }
        if (data.job) setJobs(previous => ({ ...previous, [data.job]: { status: data.type === 'progress' ? 'running' : data.type === 'job_error' ? 'error' : data.type === 'job_canceled' ? 'canceled' : 'complete', result: data.result, version:data.draft_version } }))
      }
      socket.onclose = () => { if (!stopped) { setConnection('reconnecting'); reconnect = setTimeout(connect, Math.min(8000, 500 * 2 ** attempts++)) } }
      socket.onerror = () => socket.close()
    }
    connect()
    let previousJobs = {}
    const poll = setInterval(async () => {
      try {
        const result = await api(`/projects/${id}/jobs`)
        if (active.current !== id) return
        setJobs(result)
        const changedJobs = previousJobs ? Object.entries(result).filter(([kind,value])=>JSON.stringify(value)!==JSON.stringify(previousJobs[kind])).map(([,value])=>value) : []
        if (changedJobs.length) {
          await refreshProject(id)
          const failed = changedJobs.find(job => job.status === 'error')
          if (failed) setError(failed.error)
        }
        previousJobs = result
      } catch { /* reconnect will fetch the saved project */ }
    }, 4000)
    return () => { stopped = true; clearTimeout(reconnect); clearInterval(poll); socket?.close() }
  }, [project?.id])
  const promptContext = JSON.stringify([project?.system_prompt_revision||0,(project?.system_prompts||[]).filter(p=>p.enabled).map(p=>[p.name,p.text])])
  const modelContext = JSON.stringify([runtime?.runtime.state, runtime?.runtime.model, runtime?.runtime.config, promptContext])
  const conflictContext = JSON.stringify(project?.comments?.filter(c=>c.kind==='conflict'&&c.status==='open'&&!c.outdated).map(c=>c.id)||[])
  const refineContext = modelContext + JSON.stringify([project?.settings.output,project?.documents.map(d=>d.id),project?.comments?.filter(c=>c.kind==='conflict').map(c=>[c.id,c.status,c.outdated])||[]])
  const researchContext = JSON.stringify([project?.settings.research,project?.settings.scholar,promptContext])
  useEffect(()=>{
    // Restores, accepted edits, and recovered drafts also pass through the gate.
    if(sampler.current&&sampler.current.text!==editor.text)sampler.current.observe(editor.text)
  },[project?.id,editor.text])
  useEffect(() => {
    if (!project || dialog?.kind === 'merge') return
    const id = project.id
    const timer = setInterval(() => {
      const draft=buffer.current?.snapshot(), gate=sampler.current
      if(active.current!==id||document.hidden||composing.current||busy||!gate||!draft||draft.dirty||draft.saving||draft.error||gate.text!==draft.text||project.processing?.settled?.version===draft.version)return
      const send=(kind,path,context)=>{
        // Capture this saved revision; edits after dispatch cancel obsolete work
        // on the backend. Mark synchronously so a slow HTTP response cannot
        // enqueue the same content repeatedly.
        gate.mark(kind,context)
        startJob(kind,path,{expected_version:draft.version}).catch(err=>{
          if(active.current!==id)return
          if(err.status!==409)setError(err.message)
        })
      }
      if(automatic&&runtime?.runtime.state==='ready'&&conflictContext==='[]'&&jobs.refine?.status!=='running'&&gate.decision('refine',{context:refineContext,latency:runtime.runtime.last_generation?.latency_ms||0}).ready){
        send('refine','/refine',refineContext);return
      }
      // Keep review/research on their ten-second cadence and let enrichment
      // finish first; all three consumers use the same content boundary rules.
      if(jobs.refine?.status==='running'||jobs.comments?.status==='running'||jobs.research?.status==='running')return
      if(commenting&&gate.decision('comments',{context:modelContext,interval:10000}).ready){
        send('comments','/comments/review',modelContext);return
      }
      if(project.settings.research&&project.agent!=='splash'&&gate.decision('research',{context:researchContext,interval:10000}).ready)
        send('research','/research',researchContext)
    },500)
    return ()=>clearInterval(timer)
  },[project?.id,project?.settings.research,project?.processing?.settled?.version,automatic,commenting,modelContext,refineContext,researchContext,conflictContext,dialog?.kind==='merge',busy,jobs])
  const action = (path, body = {}, method = 'POST') => guard(async () => {
    const version = await buffer.current.flush()
    const result = await api(`/projects/${active.current}${path}`, { method, body: { ...body, ...(body.expected_version === true ? { expected_version: version } : {}) } })
    if (result.id) await updateProject(result)
    return result
  })
  const settings = body => action('', body, 'PATCH')
  const selectPassage = event => {
    const node=event.currentTarget
    if(node.selectionStart===node.selectionEnd) return
    const bounds=codePointSelection(node.value,node.selectionStart,node.selectionEnd)
    setSelection({...bounds,quote:node.value.slice(node.selectionStart,node.selectionEnd),text:node.value})
  }
  const locateComment = anchor => {
    const [start,end]=inputSelection(editor.text,anchor)
    textarea.current.focus(); textarea.current.setSelectionRange(start,end)
    setSelection({...anchor,quote:editor.text.slice(start,end),text:editor.text})
    textarea.current.scrollIntoView({block:'center',behavior:'smooth'})
  }
  const createComment = text => guard(async()=>{
    const version=await buffer.current.flush()
    if(!selection || selection.text!==buffer.current.text) throw new Error('Select the passage again in your current draft.')
    const p=await api(`/projects/${project.id}/comments`,{method:'POST',body:{text,start:selection.start,end:selection.end,expected_version:version}})
    await updateProject(p); setSelectedComment(p.comments.at(-1).id); setSelection(null)
    return p
  })
  const runReview = (id=null) => guard(async()=>{
    const version=await buffer.current.flush()
    const kind=id?'comment_reply':'comments'
    if(!id)sampler.current?.mark('comments',modelContext)
    await startJob(kind,`/comments/${id?`${id}/ask`:'review?force=true'}`,{expected_version:version})
  })
  const launch = kind => guard(async () => {
    const version = await buffer.current.flush()
    const paths = { refine: '/refine?force=true', branches: '/branches/generate', synthesize: '/branches/synthesize', research: '/research?force=true' }
    const body = { expected_version: version }
    if(kind==='refine'||kind==='research')sampler.current?.mark(kind,kind==='refine'?refineContext:researchContext)
    if (kind === 'synthesize') body.ids = project.branches.filter(c => c.selected).map(c => c.id)
    await startJob(kind,paths[kind],body)
  })
  const beginNew = (source = null, revisionId = null, agent = 'splash') => guard(async () => {
    await buffer.current?.flush()
    setDialog({ kind: 'new', source, revisionId, agent: source ? AGENTS.find(a => a.id !== source.agent).id : agent })
  })
  const reviewMerge = () => guard(async () => {
    const version = await buffer.current.flush()
    if (preview.ids.length && version !== preview.version) throw new Error('The draft changed. Wait for the live preview to update before reviewing it.')
    const review = await api(`/projects/${project.id}/merge-review`, { method: 'POST', body: { expected_version: version, proposal_ids: preview.ids } })
    setDialog({ kind: 'merge', review, pid: project.id })
  })
  const closeDialog = () => {
    if (dialog?.kind === 'merge') api(`/projects/${dialog.pid}/merge/${dialog.review.id}`, { method: 'DELETE' }).catch(() => {})
    setDialog(null)
  }
  const addDocument = file => guard(async () => {
    await buffer.current.flush()
    const form = new FormData(); form.append('file', file)
    setProject(await api(`/projects/${project.id}/documents`, { method: 'POST', body: form })); setNotes('')
  })
  const main = project?.revisions.find(r => r.id === project.main_id)
  const agent = AGENTS.find(a => a.id === project?.agent)
  const selected = project?.branches.filter(c => c.selected) || []
  const running = Object.entries(jobs).filter(([, job]) => job.status === 'running').map(([name]) => name)
  const modelReady = runtime?.runtime.state === 'ready'
  const sources = ids => project.sources.filter(s => ids?.includes(s.id))
  const preview = project && !editor.dirty && !editor.saving && editor.version === project.draft.version && editor.text === project.draft.text
    ? liveEnrichment(project.draft, project.proposals) : {text:editor.text, ids:[], version:editor.version}
  const researchProposals = project?.proposals.filter(p => !isEnrichment(p)) || []
  const openConflicts = project?.comments?.filter(c=>c.kind==='conflict'&&c.status==='open'&&!c.outdated) || []
  const enrichmentStatus = running.includes('refine') ? 'Enriching your prompt…'
    : editor.dirty || editor.saving ? 'Updating from your draft…'
    : openConflicts.length ? 'Conflicting directions need your review in Document comments'
    : project?.enrichment_review && project.enrichment_review.version===editor.version && project.enrichment_review.quality.status==='needs_attention' ? 'Review needs attention · see the review report below'
    : preview.ids.length ? 'Live enrichment ready'
    : jobs.refine?.status === 'error' ? 'Enrichment failed · see the error above'
    : project?.processing?.settled?.version===editor.version ? 'Your chosen wording is saved · waiting for new input'
    : jobs.refine?.status === 'canceled' && jobs.refine.version===editor.version ? 'Review interrupted · use Refine entire prompt to retry'
    : jobs.refine?.status === 'complete' && jobs.refine.version===editor.version ? jobs.refine.reason || 'Reviewed · waiting for new input'
    : modelReady ? automatic ? sampler.current?.decision('refine',{context:refineContext}).reason || 'Waiting for a complete thought…' : 'Live enrichment paused' : 'Load a model to enrich this prompt'

  return <DialogError.Provider value={error}><div className="studio">
    <aside className="sidebar">
      <a href="/" className="brand" aria-label="Splash · Start a new prompt project" onClick={e => { e.preventDefault(); beginNew() }}><span className="brand-art"><img src={splashLogo} width="1024" height="1024" alt="Splash — Prompt Enrichment" /></span></a>
      <button className="primary new-project" onClick={() => beginNew()}>＋ New prompt project</button>
      <input className="project-search" aria-label="Search projects" placeholder="Search your projects…" value={query} onChange={e => setQuery(e.target.value)} />
      <div className="sidebar-heading">YOUR PROJECTS <span>{projects.length}</span></div>
      <nav aria-label="Prompt projects">{projects.filter(p => `${p.title} ${p.agent}`.toLowerCase().includes(query.toLowerCase())).map(p => <button key={p.id} className={`project-link ${project?.id === p.id ? 'active' : ''}`} onClick={() => guard(() => openProject(p.id))}><span>{AGENTS.find(a => a.id === p.agent)?.icon}</span><div><strong>{p.title}</strong><small>{p.agent}</small></div></button>)}{!projects.length && <p className="muted sidebar-empty">Your ideas will have a home here.</p>}</nav>
      <div className="sidebar-bottom"><button className="text-button" onClick={() => { setError(''); setDialog({ kind: 'system-prompts' }) }}>Prompt library</button><button className="text-button" onClick={() => setDialog({ kind: 'models' })}>⚙ Model & hardware</button><button className="text-button" onClick={() => setDialog({ kind: 'import-file', agent: 'splash' })}>↗ Import project archive</button><span className="local-label"><i /> Saved on this computer</span></div>
    </aside>
    <main className="main-area">
      <header className="topbar"><span className="breadcrumb">PROMPT WORKSPACE {agent && <>/ <b>{agent.id}</b></>}</span><button className="model-pill" onClick={() => setDialog({ kind: 'models' })}><i className={modelReady ? 'ready' : ''} />{modelReady ? runtime.runtime.model + (runtime.runtime.engine==='endpoint'?' · existing endpoint':' · local diffusion') : runtime?.runtime.state === 'loading' ? 'Loading model…' : 'Choose a local model'} ↗</button></header>
      {error && <div className="alert error" role="alert">{error}<button onClick={() => setError('')} aria-label="Dismiss error">×</button></div>}
      {notice && <div className="alert" role="status">{notice}<button onClick={() => setNotice('')} aria-label="Dismiss notice">×</button></div>}
      {runtime?.runtime.state === 'error' && <div className="alert error" role="alert"><span><strong>Model unavailable.</strong> {runtime.runtime.error}</span><button onClick={() => setDialog({ kind: 'models' })}>Model & hardware</button></div>}
      {runtime?.runtime.notice && <div className="alert" role="status">{runtime.runtime.notice}<button onClick={() => setDialog({ kind: 'models' })}>{runtime.endpoints?.length?'Choose model endpoint':'Model & hardware'}</button></div>}
      {!project ? <section className="welcome"><div className="welcome-hero"><div className="welcome-copy"><p className="eyebrow">GIVE YOUR IDEAS ROOM TO GROW</p><h1>A sharper prompt.<br /><span className="brand-accent">A clearer direction.</span></h1><p>Start with what you mean. Explore what it could become.<br />Keep every version that brought you here.</p></div><div className="welcome-art" aria-hidden="true"><span className="brand-art"><img src={splashLogo} width="1024" height="1024" alt="" /></span><span className="art-caption">A little clarity. A new possibility.</span></div></div><div className="agent-grid">{AGENTS.map(a => <button key={a.id} className="agent-card" onClick={() => beginNew(null, null, a.id)}><span className="agent-icon">{a.icon}</span><h3>{a.id}</h3><strong>{a.label}</strong><p>{a.description}</p><span className="card-arrow">Start a project ↗</span></button>)}</div><p className="welcome-note">One agent per project · Local history · You approve every merge</p></section> : <>
        <section className="project-heading"><div><p className="eyebrow">{agent.label.toUpperCase()}</p><h1>{project.title}</h1><p>{project.origin ? `Imported from “${project.origin.title}” · ` : ''}{project.revisions.filter(r => r.kind === 'main').length} main versions · Started {new Date(project.created).toLocaleDateString()}</p></div><div className="project-actions"><button onClick={() => { setError(''); setDialog({ kind: 'system-prompts' }) }}>System prompts</button><button onClick={() => setDialog({ kind: 'rename', title: project.title })}>Rename</button><button onClick={() => beginNew(project)}>Import as new</button><button onClick={() => setDialog({ kind: 'export' })}>Export ↗</button><button className="text-button danger" onClick={() => setDialog({ kind: 'delete' })}>Delete</button></div></section>
        <div className="collaborative-workspace"><section className="editor-shell"><div className="editor-toolbar"><span><span className="status-dot" /> Main stays safe while you explore</span><span className="muted">{connection === 'connected' ? 'Connected locally' : 'Reconnecting…'}</span></div>
          <div className="split-editor"><section className="main-pane">
            <div className="pane-label"><span>MAIN PROMPT</span><span className="tag">{showSavedMain ? `Saved version ${project.revisions.filter(r => r.kind === 'main').length}` : 'Live preview'}</span></div>
            <p className="preview-status" role="status">{showSavedMain ? 'Your last confirmed main prompt' : enrichmentStatus}</p>
            <div className={`prompt-text ${showSavedMain ? '' : 'live-prompt'}`} tabIndex={0} aria-label={showSavedMain ? 'Saved main prompt' : 'Live enriched prompt'} aria-busy={!showSavedMain && running.includes('refine')}>{showSavedMain ? main?.text : <CommentedPrompt text={preview.text} comments={project.comments||[]} selected={selectedComment} choose={setSelectedComment}/>}</div>
            <div className="pane-footer"><button className="text-button" onClick={()=>setShowSavedMain(!showSavedMain)}>{showSavedMain ? 'Show live enrichment' : 'Show saved main'}</button><button className="text-button" onClick={() => guard(async () => { await navigator.clipboard.writeText(showSavedMain ? main.text : preview.text); setNotice(showSavedMain ? 'Saved main copied.' : 'Live prompt copied.') })}>{showSavedMain ? 'Copy saved main' : 'Copy live prompt'}</button></div>
            {!showSavedMain && preview.ids.length > 0 && <div className="preview-controls"><small>Preview only · Review merge saves this text as your next main after confirmation.</small><button className="text-button" disabled={busy} onClick={()=>action('/proposals/reject',{ids:preview.ids,expected_version:true})}>Discard enrichment</button></div>}
          </section>
          <section className="draft-pane"><div className="pane-label"><label htmlFor="working-draft">WORKING DRAFT</label><span className={editor.error ? 'error-text' : 'muted'}>{editor.error ? 'Needs attention' : editor.saving ? 'Saving…' : editor.dirty ? 'Unsaved changes' : 'All changes saved'}</span></div><textarea id="working-draft" ref={textarea} readOnly={busy} value={editor.text} maxLength={80000} spellCheck onChange={e => {sampler.current.observe(e.target.value,{caret:e.target.selectionStart,committed:e.nativeEvent.inputType==='insertFromPaste'});buffer.current.edit(e.target.value);setSelection(null)}} onSelect={selectPassage} onCompositionStart={() => { composing.current = true }} onCompositionEnd={e => { composing.current = false; sampler.current.observe(e.currentTarget.value,{caret:e.currentTarget.selectionStart}) }} /><div className="pane-footer"><span>{editor.text.trim().split(/\s+/u).filter(Boolean).length} words · {editor.text.length.toLocaleString()} / 80,000</span><button className="text-button" onClick={() => guard(async () => { await navigator.clipboard.writeText(editor.text); setNotice('Working draft copied.') })}>Copy draft</button></div></section></div>
          {editor.error && <div className="alert error"><span>{editor.error.message}</span><button onClick={() => guard(async () => { const fresh = await api(`/projects/${project.id}`); buffer.current.version = fresh.draft.version; buffer.current.error = null; await buffer.current.flush() })}>Keep local draft</button><button onClick={() => guard(async () => { const fresh = await api(`/projects/${project.id}`); await updateProject(fresh, true) })}>Use saved draft</button></div>}
          {openConflicts.length>0&&<div className="comment-conflict-banner">{openConflicts.length} open {openConflicts.length===1?'conflict':'conflicts'} · Review the highlighted passages and clarify your intent in the comment rail. The agent will not choose a direction for you.</div>}
          <div className="merge-bar"><div><label className="toggle"><input type="checkbox" checked={automatic} onChange={e => setAutomatic(e.target.checked)} />Live enrichment</label><small>{modelReady ? 'AI waits for a clause or sentence, or six completed words and a longer pause. Refine manually at any time.' : 'Load a diffusion model to see live enrichment in the main pane.'}</small></div><div className="button-row"><button disabled={!modelReady || busy || running.includes('refine') || !editor.text.trim()} onClick={() => launch('refine')}>{running.includes('refine') ? 'Refining…' : '✳ Refine entire prompt'}</button><button className="primary" disabled={busy || main?.text === preview.text || !!editor.error || editor.dirty || editor.saving} onClick={reviewMerge}>{preview.ids.length ? 'Review live merge →' : 'Review draft merge →'}</button></div></div>
          <EnrichmentReport review={project.enrichment_review} version={editor.version}/>
        </section><CommentRail key={project.id} project={project} selected={selectedComment} choose={setSelectedComment} busy={busy} dirty={editor.dirty||editor.saving} modelReady={modelReady} running={running} automatic={commenting} setAutomatic={setCommenting} selection={selection} create={createComment} action={action} ask={runReview} review={()=>runReview()} locate={locateComment}/></div>
        <section className="details"><div className="detail-nav" role="tablist" aria-label="Project tools">{[...(project.agent !== 'splash' ? [['suggestions','Research suggestions',researchProposals.length]] : []), ...(project.agent === 'splash-search' ? [['branches','Research branches',project.branches.length]] : []), ['context','Context & sources',project.documents.length], ['history','Version history',project.revisions.filter(r=>r.kind==='main').length]].map(([key,label,count])=><button key={key} role="tab" aria-selected={view===key} className={view===key?'selected':''} onClick={()=>setView(key)}>{label} <span>{count}</span></button>)}<span className="job-status" aria-live="polite">{running.length ? `${running.join(', ')} in progress…` : 'Ready when you are'}</span></div>
          {view === 'suggestions' && <div className="detail-content"><div className="section-heading"><div><h2>Suggestions from research</h2><p>Review supporting evidence before adding these suggestions to your draft.</p></div><button disabled={!researchProposals.length || busy || editor.dirty} onClick={() => action('/proposals/accept', { ids: researchProposals.map(p=>p.id), expected_version:true })}>Accept all research</button></div>{contextInfo && <p className="muted">{contextInfo.total_passages} context passages · {contextInfo.excluded_passages} outside this pass · {contextInfo.context_characters_excluded} retrieved characters outside this pass · {contextInfo.sections} prompt sections</p>}{!researchProposals.length && <div className="empty-state"><span>✧</span><h3>Research suggestions will appear here</h3><p>Enable research in Context & sources. Live language enrichment appears in the main prompt pane.</p></div>}<div className="suggestion-list">{researchProposals.map(c=><article className="suggestion" key={c.id}><p className="rationale">{c.rationale}</p><div className="change-text">{c.before && <del>{c.before}</del>}{c.after && <ins>{c.after}</ins>}</div>{c.evidence && <blockquote>{c.evidence}</blockquote>}{sources(c.sources).map(s=><SourceLink key={s.id} source={s}/>)}<div className="button-row"><button disabled={editor.dirty||busy} onClick={()=>action('/proposals/reject',{ids:[c.id],expected_version:true})}>Dismiss</button><button disabled={editor.dirty||busy} onClick={()=>action('/proposals/accept',{ids:[c.id],expected_version:true})}>Accept edit</button></div></article>)}</div></div>}
          {view === 'branches' && <div className="detail-content"><div className="section-heading"><div><h2>Where could this question lead?</h2><p>Select directions to combine. Your main stays in place.</p></div><div className="button-row"><label>Cards <select value={project.settings.branch_count} onChange={e=>settings({branch_count:Number(e.target.value)})}>{[1,2,3,4,5].map(n=><option key={n}>{n}</option>)}</select></label><button disabled={!modelReady||running.includes('branches')||selected.length>0} onClick={()=>launch('branches')}>Generate branches</button></div></div><div className="branch-grid">{project.branches.map(c=><article className={`branch-card ${c.selected?'chosen':''}`} key={c.id}><label><input type="checkbox" checked={c.selected} disabled={busy||editor.dirty} onChange={e=>action('/branches/select',{expected_version:true,ids:e.target.checked?[...selected.map(x=>x.id),c.id]:selected.filter(x=>x.id!==c.id).map(x=>x.id)})}/><strong>{c.title}</strong></label><p>{c.text}</p><small>{c.rationale}</small>{sources(c.sources).map(s=><SourceLink key={s.id} source={s}/>)}</article>)}</div>{!project.branches.length&&<div className="empty-state"><span>⌘</span><h3>Follow a promising direction</h3><p>Generate up to five branches from this prompt and its evidence.</p></div>}<div className="branch-controls"><button disabled={!project.branches.length||busy} onClick={()=>action('/branches/select',{expected_version:true,ids:selected.length===project.branches.length?[]:project.branches.map(c=>c.id)})}>{selected.length&&selected.length===project.branches.length?'Clear selection':'Select all'}</button><span className="muted">{selected.length} selected · Previous drafts are preserved</span><button className="primary" disabled={!selected.length||!modelReady||running.includes('synthesize')} onClick={()=>launch('synthesize')}>{selected.length===1?'Use branch as draft':'Synthesize selected'} →</button></div></div>}
          {view === 'context' && <div className="detail-content"><div className="section-heading"><div><h2>Give your prompt some grounding</h2><p>Text, code, Markdown, and text-based PDFs · 8 MB per file</p></div><button onClick={()=>upload.current.click()}>＋ Upload context</button><input hidden ref={upload} type="file" onChange={e=>{if(e.target.files[0])addDocument(e.target.files[0]);e.target.value=''}}/></div><div className="context-compose"><input aria-label="Context title" value={noteName} onChange={e=>setNoteName(e.target.value)}/><textarea aria-label="Paste context" placeholder="Paste notes, a research brief, or code…" value={notes} onChange={e=>setNotes(e.target.value)}/><button disabled={!notes.trim()||busy} onClick={()=>addDocument(new File([notes],noteName.toLowerCase().endsWith('.pdf')?noteName+'.txt':noteName||'notes.txt',{type:'text/plain'}))}>Save context</button></div><div className="document-list">{project.documents.map(d=><div key={d.id}><span>▤ <strong>{d.name}</strong> <small>{d.pages.length} text pages</small></span><button onClick={()=>guard(async()=>setProject(await api(`/projects/${project.id}/documents/${d.id}`,{method:'DELETE'})))}>Remove</button></div>)}</div>{project.agent!=='splash'&&<div className="research-settings"><h3>Research tools</h3><label className="toggle"><input type="checkbox" checked={project.settings.research} onChange={e=>settings({research:e.target.checked})}/>Research changed prompts about every 10 seconds</label><label className="toggle"><input type="checkbox" checked={project.settings.scholar} onChange={e=>settings({scholar:e.target.checked})}/>Google Scholar · experimental, at most once per minute</label><p className="muted">Focused queries go to public providers. Uploaded documents stay local. Source-based edits require acceptance.</p><button disabled={!project.settings.research||running.includes('research')} onClick={()=>launch('research')}>Research now</button></div>}<label className="field">Output style<select value={project.settings.output} onChange={e=>settings({output:e.target.value})}><option value="prose">Polished prose</option><option value="structured">Structured Markdown</option></select></label><h3>Sources collected</h3>{project.sources.length?project.sources.map(s=><SourceLink key={s.id} source={s}/>):<p className="muted">No sources collected yet.</p>}<details><summary>Query history ({project.queries.length})</summary>{[...project.queries].reverse().map((q,i)=><p className="query" key={i}>{stamp(q.created)} · {q.tool}<br/>{q.arguments.query}</p>)}</details></div>}
          {view === 'history' && <div className="detail-content"><div className="section-heading"><div><h2>A path back to your original idea</h2><p>Restore a version as a draft, then review and merge.</p></div><label className="toggle"><input type="checkbox" checked={allHistory} onChange={e=>setAllHistory(e.target.checked)}/>Include drafts & variants</label></div><div className="history-list">{[...project.revisions].reverse().filter(r=>allHistory||r.kind==='main').map(r=><article key={r.id}><div className="history-marker">{r.kind==='main'?'●':'○'}</div><div><h3>{r.id===project.main_id?'Current main':r.kind==='main'?'Archived main':r.kind==='variant'?'Research variant':'Saved draft'} <span className="tag">{stamp(r.created)}</span></h3><p className="history-excerpt">{r.text}</p><RevisionDetails revision={r} revisions={project.revisions} sources={project.sources}/><div className="button-row"><button onClick={()=>guard(async()=>{const data=await api(`/projects/${project.id}/compare/${r.id}`);setDialog({kind:'compare',...data,revision:r})})}>Compare with main</button><button disabled={busy} onClick={()=>action('/restore',{revision_id:r.id,expected_version:true})}>Restore as draft</button><button onClick={()=>beginNew(project,r.id)}>Import into new project</button></div></div></article>)}</div></div>}
        </section><footer className="workspace-footer"><span>Live enrichment is a preview. Saved main versions change only after a confirmed merge.</span><span>{runtime?.mcp?.state==='ready'?'MCP tools connected':'Research tools reconnecting'} · Local storage</span></footer>
      </>}
    </main>
    {dialog?.kind==='new'&&<NewProjectDialog initial={dialog} busy={busy} close={closeDialog} create={data=>guard(async()=>{const p=await api('/projects',{method:'POST',body:data});setDialog(null);await refreshList();await openProject(p.id)})}/>}
    {dialog?.kind==='models'&&<ModelDialog runtime={runtime} close={closeDialog} guard={guard} refresh={refreshRuntime}/>}
    {dialog?.kind==='system-prompts'&&<SystemPromptDialog project={project} close={closeDialog} guard={guard} busy={busy} onSaved={setProject}/>}
    {dialog?.kind==='merge'&&<Dialog title="Review your next main prompt" close={closeDialog} wide><p className="muted">Stage 1 of 2 · Green is added. Red is removed. Your input draft stays unchanged during review. Confirmation saves this exact prompt as the new main, archives the previous main and input draft, and starts the next draft from it.</p><Diff parts={dialog.review.diff}/>{openConflicts.length>0&&<p className="alert">This prompt has {openConflicts.length} unresolved conflict comments. Confirm only if you intend to preserve those competing directions.</p>}{sources(dialog.review.sources).map(s=><SourceLink key={s.id} source={s}/>)}<div className="dialog-actions"><button onClick={closeDialog}>Keep editing</button><button className="primary" disabled={busy||editor.dirty||editor.version!==dialog.review.draft_version||project.main_id!==dialog.review.main_id||dialog.review.proposals?.some(c=>!project.proposals.some(p=>JSON.stringify(p)===JSON.stringify(c)))} onClick={()=>guard(async()=>{const p=await api(`/projects/${dialog.pid}/merge/${dialog.review.id}`,{method:'POST'});await updateProject(p);setDialog(null);setNotice('Draft promoted. The previous main is preserved in Version history.')})}>Confirm merge & archive prior main</button></div></Dialog>}

    {dialog?.kind==='compare'&&<Dialog title="Archived version → current main" close={closeDialog} wide><Diff parts={dialog.diff}/><div className="dialog-actions"><button onClick={closeDialog}>Close</button><button onClick={async()=>{await action('/restore',{revision_id:dialog.revision.id,expected_version:true});setDialog(null)}}>Restore archived version as draft</button></div></Dialog>}
    {dialog?.kind==='rename'&&<Dialog title="Rename project" close={closeDialog}><form onSubmit={e=>{e.preventDefault();guard(async()=>{const p=await api(`/projects/${project.id}`,{method:'PATCH',body:{title:dialog.title}});setProject(p);await refreshList();setDialog(null)})}}><label className="field">Project name<input autoFocus required maxLength={180} value={dialog.title} onChange={e=>setDialog({...dialog,title:e.target.value})}/></label><div className="dialog-actions"><button className="primary">Save name</button></div></form></Dialog>}
    {dialog?.kind==='delete'&&<Dialog title="Delete this project?" close={closeDialog}><p>This removes “{project.title}”, its context, and all local history. Independent imports remain available.</p><div className="dialog-actions"><button onClick={closeDialog}>Cancel</button><button className="danger" onClick={()=>guard(async()=>{const id=project.id;await api(`/projects/${id}`,{method:'DELETE'});buffer.current?.close();buffer.current=null;active.current=null;setProject(null);localStorage.removeItem(`splash.recovery.${id}`);localStorage.removeItem('splash.active');setDialog(null);await refreshList()})}>Delete project and history</button></div></Dialog>}
    {dialog?.kind==='export'&&<Dialog title="Take your work with you" close={closeDialog}><p className="muted">Text and Markdown contain the main prompt. A complete project archive includes drafts, history, sources, and extracted context.</p><div className="export-options">{[['txt','Main prompt · text'],['md','Main prompt · Markdown'],['json','Complete project · JSON']].map(([format,label])=><a key={format} href={`/api/projects/${project.id}/export?format=${format}`} download>{label} ↗</a>)}</div></Dialog>}
    {dialog?.kind==='import-file'&&<Dialog title="Import a project archive" close={closeDialog}><p className="muted">Start an independent project from an exported main prompt and its context.</p><label className="field">Agent<select value={dialog.agent} onChange={e=>setDialog({...dialog,agent:e.target.value})}>{AGENTS.map(a=><option key={a.id}>{a.id}</option>)}</select></label><label className="field">Splash JSON export<input type="file" accept="application/json,.json" onChange={e=>{const file=e.target.files[0];if(file)guard(async()=>{await buffer.current?.flush();const form=new FormData();form.append('file',file);const p=await api(`/projects/import-file?agent=${encodeURIComponent(dialog.agent)}`,{method:'POST',body:form});setDialog(null);await refreshList();await openProject(p.id)})}}/></label></Dialog>}
  </div></DialogError.Provider>
}
