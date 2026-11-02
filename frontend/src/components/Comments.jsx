import React, {useEffect, useRef, useState} from 'react'
import {commentSegments} from '../client'
import {SourceLink, stamp} from './StudioParts'
import './Comments.css'

export function CommentedPrompt({text, comments, selected, choose}) {
  return commentSegments(text, comments).map((part, i) => part.ids.length
    ? <button key={i} className={`comment-anchor ${part.conflict?'conflict':''} ${part.ids.includes(selected)?'active':''}`}
        aria-label={`View ${part.conflict?'conflict':'comment'} on: ${part.text}`} onClick={()=>choose(part.ids[0])}>{part.text}</button>
    : <React.Fragment key={i}>{part.text}</React.Fragment>)
}

function CommentThread({comment, selected, choose, busy, dirty, modelReady, answering, action, ask, locate, sources, agent}) {
  const [reply,setReply] = useState('')
  const ref = useRef(null)
  useEffect(()=>{if(selected) ref.current?.scrollIntoView({block:'nearest',behavior:'smooth'})},[selected])
  const open = comment.status === 'open'
  return <article ref={ref} className={`comment-thread ${comment.kind} ${selected?'selected':''}`} aria-label={`${comment.kind} comment`} onClick={()=>choose(comment.id)}>
    <div className="comment-heading"><span className="comment-avatar">{comment.origin==='user'?'Y':comment.origin==='consistency'?'!':'✳'}</span><div><strong>{comment.origin==='user'?'You':comment.origin==='consistency'?'Consistency check':agent}</strong><small>{comment.kind} · {stamp(comment.created)}</small></div><span className="tag">{comment.outdated?'Passage changed':comment.status}</span></div>
    {comment.anchors.map((a,i)=><button className="comment-quote" key={i} disabled={dirty||comment.outdated} onClick={e=>{e.stopPropagation();choose(comment.id);locate(a)}} title="Select this passage in your working draft">{comment.anchors.length>1&&<small>Passage {i+1}</small>}{a.quote||'Insertion point'}</button>)}
    {comment.messages.map((message,i)=><div className="comment-message" key={message.id}>{i>0&&<strong>{message.author==='you'?'You':message.author==='consistency'?'Consistency check':agent}</strong>}<p>{message.text}</p></div>)}
    {comment.replacement!==null&&comment.replacement!==undefined&&<div className="comment-replacement"><small>Suggested replacement</small>{comment.replacement||<em>Remove this passage</em>}</div>}
    {comment.evidence?.length>0&&<details className="comment-evidence"><summary>Supporting context ({comment.evidence.length})</summary>{comment.evidence.map((e,i)=><blockquote key={i}><small>{e.name}</small>{e.quote}</blockquote>)}</details>}
    {sources.filter(s=>comment.sources?.includes(s.id)).map(s=><SourceLink key={s.id} source={s}/>)}
    {comment.outdated&&<p className="comment-hint">{comment.outdated_reason||'The anchored text changed.'} This discussion is preserved, but its edit cannot be applied. Review the document again for current suggestions.</p>}
    {open&&comment.kind==='conflict'&&!comment.outdated&&<p className="comment-hint">Clarify the working draft, then resolve the discussion. Resolving a comment does not change the prompt.</p>}
    <div className="comment-actions">
      {open&&comment.kind==='suggestion'&&<button className="primary" disabled={busy||dirty||comment.outdated} onClick={()=>action(`/comments/${comment.id}/apply`,{expected_version:true})}>Apply to draft</button>}
      {open&&<button disabled={busy} onClick={()=>action(`/comments/${comment.id}`,{status:'resolved'},'PATCH')}>Resolve</button>}
      {open&&<button className="text-button" disabled={busy} onClick={()=>action(`/comments/${comment.id}`,{status:'dismissed'},'PATCH')}>Dismiss</button>}
      {!open&&comment.status!=='accepted'&&<button disabled={busy} onClick={()=>action(`/comments/${comment.id}`,{status:'open'},'PATCH')}>Reopen</button>}
    </div>
    {open&&<form className="comment-reply" onSubmit={async e=>{e.preventDefault();const result=await action(`/comments/${comment.id}/reply`,{text:reply});if(result)setReply('')}}>
      <textarea aria-label="Reply to comment" value={reply} maxLength={4000} onChange={e=>setReply(e.target.value)} rows={2} placeholder="Clarify your intent or reply…"/>
      <div className="comment-actions"><button disabled={busy||!reply.trim()}>Reply</button><button type="button" disabled={busy||dirty||!modelReady||answering||comment.outdated} onClick={()=>ask(comment.id)}>{answering?'Reviewer responding…':'Ask AI to respond'}</button></div>
    </form>}
  </article>
}

export function EnrichmentReport({review, version}) {
  if(!review||review.version!==version)return null
  const {context:c,quality:q}=review
  return <details className="enrichment-report" open={q.status==='needs_attention'}>
    <summary>{q.status==='needs_attention'?'Some suggestions were withheld':q.status==='no_changes'?'Review complete · no new supported edits':`Review complete · ${q.suggestions-q.questions} edits · ${q.questions} questions`}</summary>
    <p>Context limit: {c.context_limit.toLocaleString()} tokens. Up to {c.input_tokens_max.toLocaleString()} input tokens per section; {c.output_budget.toLocaleString()} output tokens, up to {c.retry_output_budget.toLocaleString()} on a retry.</p>
    <p>Used {c.used_passages} uploaded passages; {c.excluded_passages} passages and {c.uploaded_characters_excluded.toLocaleString()} uploaded characters were outside this pass. Reviewed {c.checked_characters.toLocaleString()} draft characters across {c.sections} sections.</p>
    {c.excluded_draft_characters>0&&<p>{c.excluded_draft_characters.toLocaleString()} draft characters were held while editing.</p>}
    {q.previous_preview_retained&&<p>The model did not return a complete, valid review after one retry. Previous valid suggestions remain available. Use Refine entire prompt to retry.</p>}
    {[...q.errors,...q.withheld].map((message,i)=><p className="comment-hint" key={i}>{message}</p>)}
    {q.invalid_response_samples?.length>0&&<details className="comment-evidence"><summary>Inspect the unaccepted model response</summary>{q.invalid_response_samples.map((sample,i)=><blockquote key={i}>{sample}</blockquote>)}</details>}
    <p className="comment-hint">Checks cover exact quotations, known requirements and supported literals. Review meaning and assumptions before accepting.</p>
  </details>
}

export function CommentRail({project, selected, choose, busy, dirty, modelReady, running, automatic, setAutomatic, selection, create, action, ask, review, locate}) {
  const [filter,setFilter]=useState('open'), [text,setText]=useState('')
  const comments=project.comments||[]
  const open=comments.filter(c=>c.status==='open')
  const filtered=comments.filter(c=>filter==='all'||c.status==='open')
    .sort((a,b)=>(b.kind==='conflict')-(a.kind==='conflict')||new Date(b.created)-new Date(a.created))
  useEffect(()=>{if(selected&&comments.find(c=>c.id===selected)?.status!=='open')setFilter('all')},[selected])
  return <aside className="comment-rail" aria-label="Document comments">
    <div className="comment-rail-heading"><div><p className="eyebrow">YOU + {project.agent.toUpperCase()}</p><h2>Document comments <span>{open.length}</span></h2><p>Discuss the prompt, clarify conflicts, and review suggested edits.</p><p className="comment-hint">Your decisions are remembered. Reopen a discussion or edit its passage to revisit the wording.</p></div></div>
    <div className="comment-toolbar"><label className="toggle"><input type="checkbox" checked={automatic} onChange={e=>setAutomatic(e.target.checked)}/>Automatic review</label><button disabled={busy||dirty||running.includes('comments')} onClick={review}>{running.includes('comments')?'Reviewing…':'Review document'}</button></div>
    <div className="comment-filter"><button aria-pressed={filter==='open'} onClick={()=>setFilter('open')}>Open ({open.length})</button><button aria-pressed={filter==='all'} onClick={()=>setFilter('all')}>All discussions ({comments.length})</button></div>
    {project.comment_review?.errors?.map((e,i)=><p className="comment-hint" role="status" key={i}>{e}</p>)}
    {project.comment_review?.invalid_response_samples?.length>0&&<details className="comment-evidence"><summary>Inspect the unaccepted review response</summary>{project.comment_review.invalid_response_samples.map((sample,i)=><blockquote key={i}>{sample}</blockquote>)}</details>}
    {project.comment_review?.excluded_characters>0&&<p className="comment-hint">AI review covered {project.comment_review.checked_characters.toLocaleString()} characters; {project.comment_review.excluded_characters.toLocaleString()} were outside this pass. Local pattern checks cover the full draft.</p>}
    <div className="comment-list">{filtered.map(c=><CommentThread key={c.id} comment={c} selected={selected===c.id} choose={choose} busy={busy} dirty={dirty} modelReady={modelReady} answering={running.includes('comment_reply')} action={action} ask={ask} locate={locate} sources={project.sources} agent={project.agent}/>)}{!filtered.length&&<div className="comments-empty"><span>☷</span><h3>{filter==='open'?'No open comments':'Start a conversation'}</h3><p>Select a passage in your working draft to leave a note, or ask the reviewer to look for unclear or conflicting directions.</p></div>}</div>
    <form className="new-comment" onSubmit={async e=>{e.preventDefault();const result=await create(text);if(result)setText('')}}>
      <label htmlFor="new-comment">Add your comment</label><p className="comment-hint">{selection?.quote ? `On “${selection.quote.slice(0,100)}${selection.quote.length>100?'…':''}”` : 'Select text in the working draft to anchor a comment.'}</p>
      <textarea id="new-comment" value={text} onChange={e=>setText(e.target.value)} maxLength={4000} rows={2} placeholder="Leave a note for the AI reviewer…"/>
      <button disabled={busy||dirty||!selection?.quote||!text.trim()}>Add comment</button>
    </form>
  </aside>
}
