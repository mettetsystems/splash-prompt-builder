import React, { useEffect, useRef, useState, createContext, useContext } from 'react'
import { api, safeUrl } from '../client'

export const AGENTS = [
  { id:'splash',icon:'✳',label:'A clearer expression',description:'Turn an idea into precise, thoughtful prose.' },
  { id:'splash-code',icon:'⌘',label:'A better specification',description:'Shape coding tasks, constraints, and acceptance criteria.' },
  { id:'splash-search',icon:'⌕',label:'A deeper question',description:'Frame research, follow evidence, and explore new directions.' },
]
export const stamp = value => new Date(value).toLocaleString([], { dateStyle:'medium', timeStyle:'short' })
export const DialogError = createContext('')
export function SourceLink({ source }) {
  const url=safeUrl(source.url)
  return <div className="source"><span className="tag">{source.level?.replaceAll('_',' ')}</span>{url?<a href={url} target="_blank" rel="noreferrer">{source.title} ↗</a>:<span>{source.title}</span>}<small>{source.provider}</small></div>
}
export function Diff({parts}) {
  return <div className="diff" aria-label="Prompt comparison">{parts.map((p,i)=><React.Fragment key={i}>{p.kind==='equal'?p.before:<>{p.before&&<del>{p.before}</del>}{p.after&&<ins>{p.after}</ins>}</>}</React.Fragment>)}</div>
}
export function RevisionDetails({revision, revisions, sources}) {
  const m=revision.metadata||{}
  const find=id=>revisions.find(r=>r.id===id)
  const origin=m.origin
  const parent=find(m.restored_from||m.draft_id||m.prior_draft||m.base_main)
  return <details className="revision-details"><summary>Provenance & sources</summary>
    {origin&&<p>Imported from “{origin.title}” · {origin.agent} · {stamp(origin.imported)}</p>}
    {parent&&<p>{m.restored_from?'Restored from':(m.draft_id||m.prior_draft)?'Based on draft':'Based on main'} saved {stamp(parent.created)}. That revision remains in this timeline.</p>}
    {!origin&&!parent&&!m.branches&&<p>Original project prompt or independently saved draft.</p>}
    {m.accepted_edits&&<p>{m.accepted_edits.length} edits explicitly accepted into this draft{m.accepted_edits[0]?.generation?.model?` · ${m.accepted_edits[0].generation.model}`:''}.</p>}
    {m.generation?.model&&<p>Generated with {m.generation.model_repository||m.generation.model} · {m.generation.model_source==='local'?'local checkpoint':`checkpoint ${m.generation.model_revision?.slice(0,12)||'unrecorded'}`}.</p>}
    {m.branches&&<><p>{m.branches.length} research directions combined. The full source cards are preserved below.</p>{m.branches.map(c=><div className="branch-origin" key={c.id}><strong>{c.title}</strong><p>{c.text}</p></div>)}</>}
    {sources.filter(s=>revision.sources.includes(s.id)).map(s=><SourceLink key={s.id} source={s}/>)}
  </details>
}
export function Dialog({title,close,children,wide=false}) {
  const dialog=useRef(null)
  const error=useContext(DialogError)
  useEffect(()=>{const node=dialog.current;node.showModal();return()=>node.close()},[])
  return <dialog ref={dialog} className={wide?'wide-dialog':''} onCancel={e=>{e.preventDefault();close()}}><div className="dialog-heading"><h2>{title}</h2><button className="icon-button" onClick={close} aria-label="Close dialog">×</button></div>{error&&<div className="alert error" role="alert">{error}</div>}{children}</dialog>
}
export function NewProjectDialog({initial,busy,close,create}) {
  const [agent,setAgent]=useState(initial.agent),[title,setTitle]=useState(''),[text,setText]=useState('')
  return <Dialog title={initial.source?'A fresh project, a new perspective':'Start with a spark'} close={close} wide><form onSubmit={e=>{e.preventDefault();create({agent,title,text:initial.source?'import':text,source:initial.source?.id||null,revision_id:initial.revisionId||null})}}><p className="muted">{initial.source?`Importing “${initial.source.title}” with its context and sources. Choose an agent for this new project.`:'Choose the agent that fits this prompting task.'}</p><div className="agent-grid compact">{AGENTS.map(a=><label className={`agent-card ${agent===a.id?'chosen':''}`} key={a.id}><input type="radio" name="agent" value={a.id} checked={agent===a.id} onChange={()=>setAgent(a.id)}/><span className="agent-icon">{a.icon}</span><h3>{a.id}</h3><p>{a.description}</p></label>)}</div><label className="field">Project name <span className="muted">optional</span><input value={title} onChange={e=>setTitle(e.target.value)} maxLength={180} placeholder="A name to find this idea later"/></label>{!initial.source&&<label className="field">Initial prompt<textarea autoFocus required value={text} onChange={e=>setText(e.target.value)} maxLength={80000} rows={5} placeholder="What would you like an LLM or agent to do?"/></label>}<div className="dialog-actions"><button type="button" onClick={close}>Cancel</button><button className="primary" disabled={busy||(!initial.source&&!text.trim())}>Create project →</button></div></form></Dialog>
}
export function ModelDialog({runtime,close,guard,refresh}) {
  const saved = runtime?.runtime.config || {}
  const [hardware,setHardware] = useState(null)
  const [working,setWorking] = useState(false)
  const [inspection,setInspection] = useState(null)
  const [config,setConfig] = useState({model:'compact',device:'cpu',path:'',steps:64,quality_mode:'quality',quantize:false,offload:false,
    repo:'',revision:'',...saved,source:saved.source || (saved.path ? 'local' : 'catalog')})
  useEffect(()=>{api('/hardware').then(setHardware).catch(()=>{})},[runtime?.runtime.state])
  const catalog = runtime?.catalog || []
  const selected = catalog.find(m=>m.id===config.model)
  const downloads = Object.values(runtime?.downloads || {})
  const busyDownload = downloads.some(d=>d.model===config.model && d.state==='downloading')
  const missing = [...(runtime?.dependencies?.missing || [])]
  if (config.quantize && runtime?.dependencies && !runtime.dependencies.packages.bitsandbytes) missing.push('bitsandbytes')
  const gpu = hardware?.gpus.find(g=>`cuda:${g.index}`===config.device)
  const profile = config.device==='cpu' ? 'cpu' : config.device==='mps' ? 'metal' : gpu ? (Number(gpu.capability)<7.5 ? 'nvidia-legacy' : 'nvidia') : 'auto'
  const level = {compact:'compact',dream:'balanced',llada:'extended'}[config.model]
  const setupCommand = `python${hardware?.os==='Windows'?'':'3'} scripts/setup.py --profile ${profile} --level ${level} --install-models${config.quantize?' --quantize':''}`
  const sourceValid = config.source==='local' ? !!config.path.trim() : config.source==='huggingface' ? !!config.repo.trim() : true
  const change = values => {setConfig(previous=>({...previous,...values}));setInspection(null)}
  const sourceBody = () => ({model:config.model,...(config.source==='local' ? {path:config.path} :
    config.source==='huggingface' ? {repo:config.repo,revision:config.revision} : {})})
  const run = operation => guard(async()=>{setWorking(true);try {return await operation()} finally {setWorking(false)}})
  const inspect = () => run(async()=>{
    const result = await api('/models/inspect',{method:'POST',body:sourceBody()})
    setInspection(result)
    if (config.source==='huggingface') setConfig(previous=>({...previous,repo:result.repo,revision:result.revision}))
  })
  const load = () => run(async()=>{
    const body = {...config,path:config.source==='local'?config.path:'',repo:config.source==='huggingface'?config.repo:'',revision:config.source==='huggingface'?config.revision:''}
    await api('/models/load',{method:'POST',body})
    await refresh()
  })
  return <Dialog title="Local intelligence, your hardware" close={close} wide>
    <p className="muted">One diffusion model serves every project. Choose an adapter family, then a recommended checkpoint, Hugging Face repository, or local directory.</p>
    {runtime?.runtime.error && <div className="alert error">{runtime.runtime.error}</div>}
    <div className="model-grid">{catalog.map(m=><label className={`model-card ${config.model===m.id?'chosen':''}`} key={m.id}>
      <input type="radio" name="model" disabled={working} checked={config.model===m.id} onChange={()=>change({model:m.id,path:'',repo:'',revision:'',context:m.context})}/>
      <strong>{m.name}</strong><p>Recommended checkpoint: ~{m.download_gb} GB<br/>{m.context.toLocaleString()} token ceiling</p>
      <small>{m.id==='compact'?'Lowest memory · experimental quality':m.id==='dream'?'Review constraints carefully':'NF4 measured on RTX 5090 · slower'}</small>
    </label>)}</div>
    <fieldset className="model-source" disabled={working}>
      <legend>Checkpoint source</legend>
      <div className="source-options">{[['catalog','Recommended'],['huggingface','Hugging Face'],['local','Local directory']].map(([value,label])=>
        <label key={value} className={`source-option ${config.source===value?'chosen':''}`}><input type="radio" name="checkpoint-source" value={value} checked={config.source===value} onChange={()=>change({source:value})}/>{label}</label>)}</div>
      {config.source==='catalog' && <div className="checkpoint-info"><a href={`https://huggingface.co/${selected?.repo}`} target="_blank" rel="noreferrer">{selected?.repo} ↗</a><small>Pinned revision: {selected?.revision}</small></div>}
      {config.source==='huggingface' && <>
        <label className="field">Hugging Face repository<input value={config.repo} maxLength={300} onChange={e=>change({repo:e.target.value})} placeholder={selected?.repo || 'owner/model'}/></label>
        <label className="field">Revision <span className="muted">optional · defaults to main</span><input value={config.revision} maxLength={200} onChange={e=>change({revision:e.target.value})} placeholder="Branch, tag, or commit SHA"/></label>
        <p className="muted">Paste an owner/model ID or repository URL. Checking resolves the revision to a fixed commit. Private or gated repositories use your local Hugging Face login or HF_TOKEN.</p>
      </>}
      {config.source==='local' && <><label className="field">Local checkpoint directory<input value={config.path} maxLength={4096} onChange={e=>change({path:e.target.value})} placeholder={hardware?.os==='Windows'?'C:\\models\\my-checkpoint':'~/models/my-checkpoint'}/></label><p className="muted">Point to a folder containing config.json, tokenizer files, and model weights. A Git checkout containing only code or LFS pointers needs its weights downloaded first.</p></>}
      <p className="muted">Custom checkpoints must match the selected adapter family. Loading may execute checkpoint Python code locally; use repositories you trust. GGUF and other model architectures need separate adapters.</p>
      <button disabled={working || !sourceValid} onClick={inspect}>{working?'Checking…':'Check checkpoint source'}</button>
      {inspection && <div className="checkpoint-info" role="status"><strong>Compatible {inspection.model_type} configuration</strong>
        {inspection.revision ? <><small>Commit: {inspection.revision}</small><small>Repository files: {inspection.download_gb===null?'size unavailable':`~${inspection.download_gb} GB`}</small></> : <small>{inspection.weight_files} weight files · ~{inspection.weight_gb} GB</small>}
        <small>{inspection.path}</small><small>Full compatibility is checked when the model loads.</small></div>}
    </fieldset>
    <div className="model-fields">
      <label className="field">Generation mode<select disabled={working} value={config.quality_mode} onChange={e=>change({quality_mode:e.target.value})}><option value="quality">Quality · more refinement</option><option value="speed">Speed · fewer refinement steps</option></select><small className="muted">Quality uses LLaDA's recommended editing thresholds and scales Dream/Compact steps with output length. Longer reviews take more time and memory.</small></label>
      <label className="field">Device<select disabled={working} value={config.device} onChange={e=>change({device:e.target.value,...(!e.target.value.startsWith('cuda:')?{quantize:false,offload:false}:{})})}>
        <option value="cpu">CPU · slower updates</option>{hardware?.gpus.map(g=><option key={g.index} value={`cuda:${g.index}`}>{g.name} · {(g.free_mb/1024).toFixed(1)} / {(g.vram_mb/1024).toFixed(1)} GiB free</option>)}
        {hardware?.os==='Darwin'&&<option value="mps">Apple Metal · execution check required</option>}
      </select></label>
      <label className="field">Diffusion steps<input type="number" min={16} max={512} disabled={working||config.model==='llada'} value={config.steps} onChange={e=>change({steps:Number(e.target.value)})}/>{config.model==='llada'&&<small className="muted">LLaDA uses its native schedule.</small>}</label>
      <label className="field">Context budget<input type="number" min={512} max={selected?.context||32768} disabled={working} value={config.context||selected?.context||1024} onChange={e=>change({context:Number(e.target.value)})}/></label>
    </div>
    <div className="button-row">
      <label className="toggle"><input type="checkbox" disabled={working||!config.device.startsWith('cuda:')} checked={config.quantize} onChange={e=>change({quantize:e.target.checked,offload:false})}/>NF4 quantization · CUDA</label>
      <label className="toggle"><input type="checkbox" disabled={working||!config.device.startsWith('cuda:')} checked={config.offload} onChange={e=>change({offload:e.target.checked,quantize:false})}/>CPU offload · 6 GB GPU / 24 GB RAM</label>
    </div>
    <div className="hardware-summary">
      {hardware ? <><strong>{hardware.os} · {hardware.cores} CPU threads · {hardware.ram_gb} GB RAM</strong><p>{hardware.disk_free_gb} GB disk free · {hardware.gpu_status}</p>{hardware.gpus.map(g=><small key={g.index}>{g.name}: {(g.free_mb/1024).toFixed(1)} GiB free of {(g.vram_mb/1024).toFixed(1)} GiB total. Used memory includes loaded models and runtime allocations.<br/></small>)}</> : <p>Scanning local hardware…</p>}
      <p>{missing.length ? `Missing model packages: ${missing.join(', ')}` : 'Model packages installed. Device execution is checked when you load a checkpoint.'}</p>
      <details><summary>Dependency setup & environment</summary><p>Run from the Splash project folder:</p><pre className="setup-command">{setupCommand}</pre><p>Check only: replace --install-models with --check.</p><small>App interpreter: {runtime?.dependencies?.python || 'Checking…'}</small>{Object.entries(runtime?.dependencies?.packages||{}).map(([name,version])=><small key={name}>{name}: {version||'not installed'}<br/></small>)}</details>
      <button disabled={working} onClick={()=>run(async()=>{setHardware(await api('/hardware'));await refresh()})}>Refresh hardware & dependencies</button>
    </div>
    {runtime?.runtime.qualification&&<p className="muted">Last execution check: {runtime.runtime.qualification.latency_ms} ms · {runtime.runtime.qualification.peak_vram_mb??'—'} MB peak GPU allocation. This does not guarantee enrichment quality.</p>}
    {downloads.length>0&&<details className="download-library"><summary>Downloaded checkpoints & transfers ({downloads.length})</summary>{downloads.map(d=><div className="checkpoint-info" key={d.id||d.model}>
      <strong>{d.repo||d.model}</strong><small>{d.stage||d.state}{d.state==='error'?`: ${d.error}`:''}</small>{d.revision&&<small>Commit: {d.revision}</small>}{d.path&&<small>{d.path}</small>}
      {d.state==='complete'&&<button disabled={working} onClick={()=>change({model:d.model,source:d.source||'catalog',repo:d.repo||'',revision:d.revision||'',path:'',context:catalog.find(m=>m.id===d.model)?.context})}>Use this checkpoint</button>}
    </div>)}</details>}
    <div className="dialog-actions">
      {config.source!=='local'&&<button disabled={working||busyDownload||!sourceValid} onClick={()=>run(async()=>{await api('/models/download',{method:'POST',body:sourceBody()});await refresh()})}>{busyDownload?'Downloading…':config.source==='huggingface'?'Download from Hugging Face':'Download recommended checkpoint'}</button>}
      <button className="primary" disabled={working||runtime?.runtime.state==='loading'||!sourceValid||missing.length>0||busyDownload} onClick={load}>{runtime?.runtime.state==='loading'?'Loading & checking…':'Load & run check'}</button>
    </div>
    <small className="muted">Data location: {runtime?.data_directory}</small>
  </Dialog>
}
