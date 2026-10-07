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
    {m.generation?.model&&<p>Generated with {m.generation.model} · checkpoint {m.generation.model_revision?.slice(0,12)}.</p>}
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
  const [hardware,setHardware]=useState(null)
  const [config,setConfig]=useState({model:'compact',device:'cpu',path:'',steps:64,quantize:false,offload:false,...runtime?.runtime.config})
  useEffect(()=>{api('/hardware').then(setHardware).catch(()=>{})},[])
  return <Dialog title="Local intelligence, your hardware" close={close} wide><p className="muted">One diffusion model serves every project. Loading runs a real generation check. Compact models may need substantial review.</p>{runtime?.runtime.error&&<div className="alert error">{runtime.runtime.error}</div>}<div className="model-grid">{runtime?.catalog.map(m=><label className={`model-card ${config.model===m.id?'chosen':''}`} key={m.id}><input type="radio" name="model" checked={config.model===m.id} onChange={()=>setConfig({...config,model:m.id,path:'',context:m.context})}/><strong>{m.name}</strong><p>~{m.download_gb} GB download · {m.context.toLocaleString()} token budget</p><small>{m.id==='compact'?'Experimental quality · lowest memory':m.id==='dream'?'Better enrichment · review constraints':'NF4 measured on RTX 5090 · slower'}<br/>Revision {m.revision.slice(0,12)}</small></label>)}</div><div className="model-fields"><label className="field">Device<select value={config.device} onChange={e=>setConfig({...config,device:e.target.value})}><option value="cpu">CPU · slower updates</option>{hardware?.gpus.map(g=><option key={g.index} value={`cuda:${g.index}`}>{g.name} · {Math.round(g.free_mb/1024)} GB free</option>)}{hardware?.os==='Darwin'&&<option value="mps">Apple Metal · execution check required</option>}</select></label><label className="field">Diffusion steps<input type="number" min={16} max={512} value={config.steps} onChange={e=>setConfig({...config,steps:Number(e.target.value)})}/></label><label className="field">Context budget<input type="number" min={512} max={runtime?.catalog.find(m=>m.id===config.model)?.context||32768} value={config.context||runtime?.catalog.find(m=>m.id===config.model)?.context||1024} onChange={e=>setConfig({...config,context:Number(e.target.value)})}/></label></div><label className="field">Existing local checkpoint directory <span className="muted">optional</span><input value={config.path} onChange={e=>setConfig({...config,path:e.target.value})} placeholder="Leave empty to use the downloaded checkpoint"/></label><div className="button-row"><label className="toggle"><input type="checkbox" checked={config.quantize} onChange={e=>setConfig({...config,quantize:e.target.checked,offload:false})}/>NF4 quantization · CUDA</label><label className="toggle"><input type="checkbox" checked={config.offload} onChange={e=>setConfig({...config,offload:e.target.checked,quantize:false})}/>CPU offload · 6 GB GPU / 24 GB RAM</label></div><div className="hardware-summary">{hardware?<><strong>{hardware.os} · {hardware.cores} CPU threads · {hardware.ram_gb} GB RAM</strong><p>{hardware.disk_free_gb} GB disk free · {hardware.gpu_status}</p>{hardware.profiles?.map(p=><small key={p.device}>{p.label}: {p.runtime||p.device}<br/></small>)}</>:<p>Scanning local hardware…</p>}<button onClick={()=>guard(async()=>setHardware(await api('/hardware')))}>Scan hardware</button></div>{runtime?.runtime.qualification&&<p className="muted">Execution check: {runtime.runtime.qualification.latency_ms} ms · {runtime.runtime.qualification.peak_vram_mb??'—'} MB peak GPU allocation. Execution success does not guarantee enrichment quality.</p>}{Object.entries(runtime?.downloads||{}).map(([key,d])=><p key={key}>{key}: {d.state} {d.error||''}</p>)}<div className="dialog-actions"><button disabled={runtime?.downloads[config.model]?.state==='downloading'} onClick={()=>guard(async()=>{await api('/models/download',{method:'POST',body:{model:config.model}});await refresh()})}>Download pinned checkpoint</button><button className="primary" disabled={runtime?.runtime.state==='loading'} onClick={()=>guard(async()=>{await api('/models/load',{method:'POST',body:config});await refresh()})}>Load & run check</button></div><small className="muted">Data location: {runtime?.data_directory}</small></Dialog>
}
