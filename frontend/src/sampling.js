// Content boundaries are a scheduling heuristic, not a grammar or truth checker.
// Saving still happens on every edit; only automatic model/tool requests wait.
const words = new Intl.Segmenter(undefined, {granularity:'word'})
const normalize = text => text.trim().replace(/\s+/gu, ' ')
function wordCount(text) {
  let count=0
  for(const part of words.segment(text))if(part.isWordLike&&++count===6)break
  return count
}
const unfinishedEnding = /\b(?:a|an|the|and|or|but|because|if|when|while|although|that|which|to|of|for|with|without|from|into|than|is|are|was|were|be|will|would|should|must|can|could)\s*$/iu

function changedWords(before, after) {
  let start=0, end=after.length, priorEnd=before.length
  while(start<Math.min(before.length,after.length)&&before[start]===after[start])start++
  while(end>start&&priorEnd>start&&after[end-1]===before[priorEnd-1]){end--;priorEnd--}
  // Include the edited word, not merely the changed letters of it.
  while(start>0&&/[\p{L}\p{N}\p{M}_]/u.test(after[start-1]))start--
  while(end<after.length&&/[\p{L}\p{N}\p{M}_]/u.test(after[end]))end++
  return wordCount(after.slice(start,end))
}

export function contentBoundary(text, previous='', {caret=text.length, committed=false}={}) {
  if(!normalize(text))return {ready:false, reason:'Write a clause or sentence to begin.'}
  if(normalize(text)===normalize(previous))return {ready:false, reason:'Waiting for new content.'}
  const prefix=text.slice(0,caret)
  // A paused cursor inside a token must not cause inference, even if the rest
  // of the document already contains many complete sentences.
  const sentence=/[.!?。！？؟۔]["'”’»\])}]*\s*$/u.test(prefix)
  const clause=/[;,；，،؛]\s*$/u.test(prefix)
  const newline=/\n\s*$/u.test(prefix)
  const space=/\s$/u.test(prefix)
  if(!sentence&&!clause&&!newline&&!space&&!committed)
    return {ready:false, reason:'Waiting for a complete word and a clause or sentence boundary.'}
  // Count words in the current unit, excluding earlier finished sentences.
  const withoutEnd=prefix.replace(/[\s.!?。！？؟۔;,；，،؛"'”’»\])}]+$/gu,'')
  const unit=withoutEnd.split(/(?:[.!?;,。！？؟۔；，،؛]["'”’»\])}]*\s+|\n+)/u).at(-1)
  const count=wordCount(unit)
  if((sentence||clause||newline)&&count>=2)
    return {ready:true, delay:2000, kind:'clause', reason:'Waiting for a pause after your clause…'}
  // Without punctuation, batch at least six completed words since the last
  // sample. A space alone after each new word does not repeatedly launch work.
  if(count>=6&&changedWords(previous,text)>=6&&!unfinishedEnding.test(unit))
    return {ready:true, delay:3500, kind:'phrase', reason:'Waiting for a pause after your phrase…'}
  return {ready:false, reason:'Waiting for a clause, sentence, or six completed words.'}
}

export class ContentSampler {
  constructor(text='', now=Date.now()) {
    this.text=text; this.caret=text.length; this.editedAt=now; this.committed=false
    this.samples=new Map(); this.requests=new Map([['comments',now],['research',now]]); this.boundaries=new Map()
  }
  observe(text, {caret=text.length, committed=false}={}, now=Date.now()) {
    this.text=text; this.caret=caret; this.committed=committed; this.editedAt=now
    this.boundaries.clear()
  }
  decision(kind, {now=Date.now(), latency=0, interval=0, context=''}={}) {
    const sample=this.samples.get(kind)
    const previous=sample?.context===context ? sample.text : ''
    let cached=this.boundaries.get(kind)
    if(!cached||cached.context!==context){
      cached={context,boundary:contentBoundary(this.text,previous,{caret:this.caret,committed:this.committed})}
      this.boundaries.set(kind,cached)
    }
    const boundary=cached.boundary
    const wait=Math.max(boundary.delay||0, Number.isFinite(latency)?latency:0)
    return {...boundary, ready:boundary.ready&&now-this.editedAt>=wait&&now-(this.requests.get(kind)??-Infinity)>=interval}
  }
  mark(kind, context='', now=Date.now()) {
    this.samples.set(kind,{text:this.text,context});this.requests.set(kind,now);this.boundaries.delete(kind)
  }
}
