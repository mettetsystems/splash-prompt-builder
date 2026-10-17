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
function promptFile(prompt) {
  return {format:'splash-system-prompt',version:1,prompt:{name:prompt.name,text:prompt.text,agent:prompt.agent||'any',note:prompt.note||''}}
}
function downloadPrompt(prompt) {
  const blob=new Blob([JSON.stringify(promptFile(prompt),null,2)],{type:'application/json'})
  const url=URL.createObjectURL(blob)
  const link=document.createElement('a')
  const slug=prompt.name.toLowerCase().replace(/[^a-z0-9]+/g,'-').replace(/^-|-$/g,'').slice(0,40)||'prompt'
  link.href=url
  link.download=`splash-system-prompt-${slug}.json`
  document.body.appendChild(link)
  link.click()
  link.remove()
  URL.revokeObjectURL(url)
}
export function SystemPromptDialog({project,close,guard,busy,onSaved}) {
  const [prompts,setPrompts]=useState(project?.system_prompts||[])
  const [name,setName]=useState('')
  const [text,setText]=useState('')
  const [library,setLibrary]=useState([])
  const [libraryReady,setLibraryReady]=useState(false)
  const [filter,setFilter]=useState('all')
  const file=useRef(null)
  const refreshLibrary=()=>api('/prompt-repo').then(data=>{setLibrary(data.prompts||[]);setLibraryReady(true)})
  useEffect(()=>{setPrompts(project?.system_prompts||[]);setName('');setText('')},[project?.id])
  useEffect(()=>{let live=true;api('/prompt-repo').then(data=>{if(!live)return;setLibrary(data.prompts||[]);setLibraryReady(true)}).catch(()=>{if(live)setLibraryReady(true)});return()=>{live=false}},[])
  const incomplete=prompts.some(prompt=>!prompt.name.trim()||!prompt.text.trim())
  const persist=next=>guard(async()=>{
    const saved=await api(`/projects/${project.id}/system-prompts`,{method:'PUT',body:{prompts:next.map(({id,name:label,text:body,enabled})=>({id:id||'',name:label,text:body,enabled:!!enabled}))}})
    onSaved(saved)
    setPrompts(saved.system_prompts||[])
    return saved
  })
  const update=(id,patch)=>setPrompts(prompts.map(item=>item.id===id?{...item,...patch}:item))
  const publish=item=>guard(async()=>{
    await api('/prompt-repo',{method:'POST',body:{name:item.name.trim(),text:item.text.trim(),agent:item.agent||project?.agent||'any',note:(item.note||'').trim()}})
    await refreshLibrary()
  })
  const enabled=prompts.filter(item=>item.enabled&&item.name.trim()&&item.text.trim())
  const fallback=library.find(item=>item.builtin&&item.agent===project?.agent)
  const combined=enabled.map(item=>item.text.trim()).join('\n\n')
  const running=project?(enabled.length?{name:(enabled.length===1?enabled[0].name:`${project.title} prompts`).trim().slice(0,80),text:combined,agent:project.agent,note:`Exported from “${project.title}”.`.slice(0,240),custom:true,tooLong:combined.length>2000}:fallback?{...fallback,custom:false,tooLong:false}:null):null
  const visible=[...library].filter(item=>filter==='all'||item.agent===filter||item.agent==='any').sort((a,b)=>(a.builtin?0:1)-(b.builtin?0:1)||(project&&a.agent===project.agent?0:1)-(project&&b.agent===project.agent?0:1))
  return <Dialog title={project?'System prompts':'Prompt library'} close={close} wide>
    {project&&<>
      <p className="muted">{running?.custom?`Enabled prompts on “${project.title}” replace the ${project.agent} default. They stay with this project, and another project does not receive them.`:`“${project.title}” has no enabled system prompt, so ${running?.name||`the ${project.agent} default`} is sent with the model. Pull or write one when you want this project to differ.`}</p>
      <div className="prompt-card-foot"><span className="muted">{running?.tooLong?'The combined running prompt is over 2,000 characters. Save each prompt to the library on its own.':'Built-in agent rules still apply. Nothing is merged until you accept it.'}</span><span className="library-actions"><button type="button" disabled={busy||!running?.custom||running.tooLong} onClick={()=>publish(running)}>Save running prompt</button><button type="button" disabled={!running} onClick={()=>downloadPrompt(running)}>Download running prompt</button></span></div>
      <ul className="prompt-list">{prompts.map(prompt=><li className="prompt-card" key={prompt.id||prompt.name}><div className="prompt-card-head"><label className="toggle"><input type="checkbox" checked={!!prompt.enabled} disabled={busy} onChange={e=>{const next=prompts.map(item=>item.id===prompt.id?{...item,enabled:e.target.checked}:item);setPrompts(next);persist(next)}}/>Use when the model runs</label><button type="button" className="text-button danger" disabled={busy} onClick={()=>persist(prompts.filter(item=>item.id!==prompt.id))}>Remove</button></div><label className="field">Name<input value={prompt.name} maxLength={80} onChange={e=>update(prompt.id,{name:e.target.value})}/></label><label className="field">Instruction<textarea rows={4} maxLength={2000} value={prompt.text} onChange={e=>update(prompt.id,{text:e.target.value})}/></label><div className="prompt-card-foot"><span className="muted">{prompt.text.length.toLocaleString()} / 2,000</span><span className="library-actions"><button type="button" disabled={busy||!prompt.name.trim()||!prompt.text.trim()} onClick={()=>publish({...prompt,agent:project.agent,note:`Saved from “${project.title}”.`})}>Save to library</button><button type="button" disabled={busy||incomplete||!prompt.name.trim()||!prompt.text.trim()} onClick={()=>persist(prompts)}>Save changes</button></span></div></li>)}{!prompts.length&&<li className="muted">No system prompt on this project yet. The {project.agent} default runs until you add one.</li>}</ul>
      <form onSubmit={e=>{e.preventDefault();persist([...prompts,{id:'',name:name.trim(),text:text.trim(),enabled:true}]).then(saved=>{if(saved){setName('');setText('')}})}}>
        <h3>Add a system prompt</h3>
        <label className="field">Name<input value={name} maxLength={80} onChange={e=>setName(e.target.value)} placeholder="A short label, such as Voice"/></label>
        <label className="field">Instruction<textarea value={text} maxLength={2000} rows={5} onChange={e=>setText(e.target.value)} placeholder="Prefer concrete examples. Keep the rewrite shorter than the draft."/></label>
        <div className="dialog-actions"><span className="muted">{text.length.toLocaleString()} / 2,000 · {prompts.length} / 12 saved</span><button className="primary" disabled={busy||prompts.length>=12||incomplete||!name.trim()||!text.trim()}>Add system prompt</button></div>
      </form>
    </>}
    <div className="library-heading"><h3>Prompt library</h3><div className="library-actions"><label className="field">Show<select aria-label="Library filter" value={filter} onChange={e=>setFilter(e.target.value)}><option value="all">All modes</option>{AGENTS.map(agent=><option key={agent.id} value={agent.id}>{agent.id}</option>)}</select></label><button type="button" onClick={()=>file.current?.click()}>Import JSON</button><input hidden ref={file} type="file" accept="application/json,.json" onChange={e=>{const chosen=e.target.files?.[0];e.target.value='';if(!chosen)return;guard(async()=>{let data;try{data=JSON.parse(await chosen.text())}catch{throw new Error('Choose a Splash system prompt JSON file.')};await api('/prompt-repo/import',{method:'POST',body:data});await refreshLibrary()})}}/></div></div>
    {!project&&<p className="muted">Saved prompts stay on this computer. Open a project when you want to pull one into it. Each mode also has a built-in default that runs until a project defines its own.</p>}
    <ul className="prompt-list">{visible.map(item=>{const inUse=project&&!running?.custom&&item.builtin&&item.agent===project.agent;return <li className="prompt-card" key={item.id}><div className="prompt-card-head"><strong>{item.name}</strong><span className="prompt-meta">{inUse&&<span className="tag">In use</span>}{item.builtin&&<span className="tag">Built in</span>}<span className="tag">{item.agent}</span></span></div>{item.note&&<p className="muted">{item.note}</p>}<p className="library-text">{item.text}</p><div className="library-actions"><button type="button" disabled={!project||busy||prompts.length>=12||incomplete} onClick={()=>persist([...prompts,{id:'',name:item.name,text:item.text,enabled:true}])}>Pull into this project</button><button type="button" onClick={()=>downloadPrompt(item)}>Download</button>{!item.builtin&&<button type="button" className="text-button danger" disabled={busy} onClick={()=>guard(async()=>{const data=await api(`/prompt-repo/${item.id}`,{method:'DELETE'});setLibrary(data.prompts||[])})}>Remove</button>}</div></li>})}{!libraryReady&&<li className="muted">Loading the library…</li>}{libraryReady&&!visible.length&&<li className="muted">No prompts in this view yet.</li>}</ul>
  </Dialog>
}
export function ModelDialog({runtime,close,guard,refresh}) {
  const [hardware,setHardware]=useState(null)
  const [config,setConfig]=useState({model:'compact',device:'cpu',path:'',steps:64,quantize:false,offload:false,...runtime?.runtime.config})
  useEffect(()=>{api('/hardware').then(setHardware).catch(()=>{})},[])
  return <Dialog title="Local intelligence, your hardware" close={close} wide><p className="muted">One diffusion model serves every project. Loading runs a real generation check. Compact models may need substantial review.</p>{runtime?.runtime.error&&<div className="alert error">{runtime.runtime.error}</div>}<div className="model-grid">{runtime?.catalog.map(m=><label className={`model-card ${config.model===m.id?'chosen':''}`} key={m.id}><input type="radio" name="model" checked={config.model===m.id} onChange={()=>setConfig({...config,model:m.id,path:'',context:m.context})}/><strong>{m.name}</strong><p>~{m.download_gb} GB download · {m.context.toLocaleString()} token budget</p><small>{m.id==='compact'?'Experimental quality · lowest memory':m.id==='dream'?'Better enrichment · review constraints':'NF4 measured on RTX 5090 · slower'}<br/>Revision {m.revision.slice(0,12)}</small></label>)}</div><div className="model-fields"><label className="field">Device<select value={config.device} onChange={e=>setConfig({...config,device:e.target.value})}><option value="cpu">CPU · slower updates</option>{hardware?.gpus.map(g=><option key={g.index} value={`cuda:${g.index}`}>{g.name} · {Math.round(g.free_mb/1024)} GB free</option>)}{hardware?.os==='Darwin'&&<option value="mps">Apple Metal · execution check required</option>}</select></label><label className="field">Diffusion steps<input type="number" min={16} max={512} value={config.steps} onChange={e=>setConfig({...config,steps:Number(e.target.value)})}/></label><label className="field">Context budget<input type="number" min={512} max={runtime?.catalog.find(m=>m.id===config.model)?.context||32768} value={config.context||runtime?.catalog.find(m=>m.id===config.model)?.context||1024} onChange={e=>setConfig({...config,context:Number(e.target.value)})}/></label></div><label className="field">Existing local checkpoint directory <span className="muted">optional</span><input value={config.path} onChange={e=>setConfig({...config,path:e.target.value})} placeholder="Leave empty to use the downloaded checkpoint"/></label><div className="button-row"><label className="toggle"><input type="checkbox" checked={config.quantize} onChange={e=>setConfig({...config,quantize:e.target.checked,offload:false})}/>NF4 quantization · CUDA</label><label className="toggle"><input type="checkbox" checked={config.offload} onChange={e=>setConfig({...config,offload:e.target.checked,quantize:false})}/>CPU offload · 6 GB GPU / 24 GB RAM</label></div><div className="hardware-summary">{hardware?<><strong>{hardware.os} · {hardware.cores} CPU threads · {hardware.ram_gb} GB RAM</strong><p>{hardware.disk_free_gb} GB disk free · {hardware.gpu_status}</p>{hardware.profiles?.map(p=><small key={p.device}>{p.label}: {p.runtime||p.device}<br/></small>)}</>:<p>Scanning local hardware…</p>}<button onClick={()=>guard(async()=>setHardware(await api('/hardware')))}>Scan hardware</button></div>{runtime?.runtime.qualification&&<p className="muted">Execution check: {runtime.runtime.qualification.latency_ms} ms · {runtime.runtime.qualification.peak_vram_mb??'—'} MB peak GPU allocation. Execution success does not guarantee enrichment quality.</p>}{Object.entries(runtime?.downloads||{}).map(([key,d])=><p key={key}>{key}: {d.state} {d.error||''}</p>)}<div className="dialog-actions"><button disabled={runtime?.downloads[config.model]?.state==='downloading'} onClick={()=>guard(async()=>{await api('/models/download',{method:'POST',body:{model:config.model}});await refresh()})}>Download pinned checkpoint</button><button className="primary" disabled={runtime?.runtime.state==='loading'} onClick={()=>guard(async()=>{await api('/models/load',{method:'POST',body:config});await refresh()})}>Load & run check</button></div><small className="muted">Data location: {runtime?.data_directory}</small></Dialog>
}
