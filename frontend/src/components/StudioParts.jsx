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
  const endpointActive = ['endpoint','ollama'].includes(runtime?.runtime.config?.source)
  const saved = endpointActive ? {} : runtime?.runtime.config || {}
  const [endpoints,setEndpoints] = useState(runtime?.endpoints || [])
  const [endpointUrl,setEndpointUrl] = useState(endpointActive ? runtime.runtime.config.base_url : '')
  const [endpointContext,setEndpointContext] = useState(endpointActive ? runtime.runtime.config.context : 4096)
  const [separateModel,setSeparateModel] = useState(false)
  useEffect(()=>setEndpoints(runtime?.endpoints||[]),[runtime?.endpoints])
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
    const body = {...config,path:config.source==='local'?config.path:'',repo:config.source==='huggingface'?config.repo:'',revision:config.source==='huggingface'?config.revision:'',confirm_separate_model:separateModel}
    try { await api('/models/load',{method:'POST',body}) } finally { await refresh() }
  })
  return <Dialog title="Local intelligence, your hardware" close={close} wide>
    <OllamaControls guard={guard} refresh={refresh} runtime={runtime}/>
    <section className="model-source">
      <h3>Use an existing model endpoint</h3>
      <p>Already running Project Maya, vLLM, llama.cpp, or another OpenAI-compatible server? Reuse its model without loading a second set of weights in Splash.</p>
      {endpointActive&&<p role="status">Connected endpoint: {runtime.runtime.config.base_url} · {runtime.runtime.model}</p>}
      <label className="field">Local API URL<input value={endpointUrl} onChange={e=>setEndpointUrl(e.target.value)} placeholder="http://127.0.0.1:8000/v1" disabled={working}/></label>
      <button disabled={working||runtime?.runtime.state==='loading'} onClick={()=>run(async()=>{const result=await api('/models/endpoints/scan',{method:'POST',body:{base_url:endpointUrl}});setEndpoints(result.endpoints);await refresh()})}>Check endpoint / scan common ports</button>
      <label className="field">Endpoint context budget<input type="number" min={1024} max={32768} value={endpointContext} onChange={e=>setEndpointContext(Number(e.target.value))} disabled={working}/></label>
      <p className="muted">Leave the URL empty to scan common localhost ports. Detection only reads model lists. Requests use the existing server's memory and capacity. Authentication uses SPLASH_ENDPOINT_API_KEY in the server environment.</p>
      {(endpoints.length?endpoints:runtime?.endpoints||[]).map(endpoint=><div className="checkpoint-info" key={endpoint.base_url}>
        <strong>{endpoint.base_url} · {endpoint.provider}</strong>
        <small>{endpoint.loaded_only?'Only currently loaded language models are offered.':endpoint.notice}</small>
        {!endpoint.models.length&&<p>No usable models listed. Start a model in that server or configure its API key, then check again.</p>}
        {endpoint.models.map(model=><button key={model.id} disabled={working||runtime?.runtime.state==='loading'} onClick={()=>run(async()=>{await api('/models/endpoint',{method:'POST',body:{base_url:endpoint.base_url,model:model.id,context:endpointContext}});await refresh()})}>Use existing endpoint · {model.id}</button>)}
      </div>)}
    </section>
    <h3>Or load a separate local diffusion model</h3>
    <p className="muted">Choose an adapter family and checkpoint. This allocates additional RAM/VRAM; the free-memory check still applies.</p>
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
    {runtime?.runtime.qualification&&<p className="muted">{endpointActive?'Endpoint connectivity verified without generation. Context budgeting uses a conservative byte estimate.':`Last execution check: ${runtime.runtime.qualification.latency_ms} ms · ${runtime.runtime.qualification.peak_vram_mb??'—'} MB peak GPU allocation.`} This does not guarantee enrichment quality.</p>}
    {downloads.length>0&&<details className="download-library"><summary>Downloaded checkpoints & transfers ({downloads.length})</summary>{downloads.map(d=><div className="checkpoint-info" key={d.id||d.model}>
      <strong>{d.repo||d.model}</strong><small>{d.stage||d.state}{d.state==='error'?`: ${d.error}`:''}</small>{d.revision&&<small>Commit: {d.revision}</small>}{d.path&&<small>{d.path}</small>}
      {d.state==='complete'&&<button disabled={working} onClick={()=>change({model:d.model,source:d.source||'catalog',repo:d.repo||'',revision:d.revision||'',path:'',context:catalog.find(m=>m.id===d.model)?.context})}>Use this checkpoint</button>}
    </div>)}</details>}
    <div className="dialog-actions">
      <label className="toggle"><input type="checkbox" disabled={working} checked={separateModel} onChange={e=>setSeparateModel(e.target.checked)}/>I want to load a separate model even if another server is running.</label>
      {config.source!=='local'&&<button disabled={working||busyDownload||!sourceValid} onClick={()=>run(async()=>{await api('/models/download',{method:'POST',body:sourceBody()});await refresh()})}>{busyDownload?'Downloading…':config.source==='huggingface'?'Download from Hugging Face':'Download recommended checkpoint'}</button>}
      <button className="primary" disabled={working||runtime?.runtime.state==='loading'||!sourceValid||missing.length>0||busyDownload} onClick={load}>{runtime?.runtime.state==='loading'?'Loading & checking…':'Load & run check'}</button>
    </div>
    <small className="muted">Data location: {runtime?.data_directory}</small>
  </Dialog>
}

function OllamaControls({guard,refresh,runtime}) {
  const [info,setInfo]=useState(null), [model,setModel]=useState(''), [confirmed,setConfirmed]=useState(false), [busy,setBusy]=useState(false)
  const check=async()=>{const value=await api('/models/ollama');setInfo(value)}
  const run=operation=>guard(async()=>{setBusy(true);try{await operation();await refresh();await check()}finally{setBusy(false)}})
  return <section className="model-source"><h3>Ollama catalog & local indexing</h3>
    <p>Check local Ollama for installed models and a recommendation based on free memory. Downloading saves weights; starting a model allocates memory.</p>
    <button disabled={busy} onClick={()=>run(check)}>Check Ollama & available memory</button>
    {info&&<><p role="status">{info.ollama.available?'Ollama is available.':info.ollama.error} {info.recommendation_error}</p>
      <label className="field">Model<select value={model} onChange={e=>setModel(e.target.value)}><option value="">Recommended: {info.recommended_model||'none fits'}</option>{info.catalog.map(item=><option key={item.id} value={item.id}>{item.name} · {item.ram_gb} GB estimated memory{item.downloaded?' · downloaded':''}{item.compatible?'':' · insufficient free memory'}</option>)}</select></label>
      <button disabled={busy||!(model||info.recommended_model)} onClick={()=>run(()=>api('/models/ollama/download',{method:'POST',body:{model:model||info.recommended_model}}))}>Download selected model</button>
      <button disabled={busy} onClick={()=>run(()=>api('/models/ollama/download',{method:'POST',body:{model:info.ollama.embedding_model}}))}>Download EmbeddingGemma for indexing</button>
      <label className="toggle"><input type="checkbox" checked={confirmed} onChange={e=>setConfirmed(e.target.checked)}/>I want to start this model in Ollama. Other AI servers may already use memory.</label>
      <button disabled={busy||!confirmed||!(model||info.recommended_model)||runtime?.runtime.state==='loading'} onClick={()=>run(()=>api('/models/ollama/load',{method:'POST',body:{model:model||info.recommended_model,automatic:!model,confirm_separate_model:true}}))}>Start selected model</button>
      <p className="muted">Use the existing-endpoint controls below to reuse a loaded model. Memory estimates cannot guarantee that every context length fits.</p>
    </>}
  </section>
}
