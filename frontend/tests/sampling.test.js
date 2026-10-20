import test from 'node:test'
import assert from 'node:assert/strict'
import {ContentSampler, contentBoundary} from '../src/sampling.js'

test('letters, individual words, and unfinished words never become eligible just by waiting',()=>{
  for(const text of ['a','p','prompt','prompt ','Build a useful prompt with cle','Use Python. Add a']){
    const sampler=new ContentSampler(text,0)
    assert.equal(sampler.decision('refine',{now:60000}).ready,false,text)
  }
})

test('a short statement or clause needs a boundary and a pause',()=>{
  for(const text of ['Use Python.','The budget is $500;','The app works offline, ','Include tests\n','请帮我写一个研究计划。','استخدم لغة بايثون؛']){
    const sampler=new ContentSampler(text,0)
    assert.equal(sampler.decision('refine',{now:1999}).ready,false,text)
    assert.equal(sampler.decision('refine',{now:2000}).ready,true,text)
  }
  assert.equal(contentBoundary('Build reliable software and ').ready,false)
})

test('an unpunctuated phrase requires six completed words and a longer pause',()=>{
  const sampler=new ContentSampler('Build a useful prompt with clear ',0)
  assert.equal(sampler.decision('refine',{now:3499}).ready,false)
  assert.equal(sampler.decision('refine',{now:3500}).ready,true)
  sampler.mark('refine','',3500)
  sampler.observe('Build a useful prompt with clear goals ',{},4000)
  assert.equal(sampler.decision('refine',{now:60000}).ready,false)
  sampler.observe('Build a useful prompt with clear goals and carefully stated acceptance criteria ',{},61000)
  assert.equal(sampler.decision('refine',{now:64500}).ready,true)
  assert.equal(contentBoundary('I would like to create a ').ready,false)
})

test('finishing an earlier sentence does not expose the unfinished next unit',()=>{
  assert.equal(contentBoundary('Use Python. Next ').ready,false)
  assert.equal(contentBoundary('Use Python. Next build ').ready,false)
  assert.equal(contentBoundary('Use Python. Include tests.').ready,true)
})

test('edits in the middle of an existing long document still wait for a word boundary',()=>{
  const text='Build a useful prompt with Python and detailed acceptance criteria.'
  const caret=text.indexOf('Python')+2
  const sampler=new ContentSampler(text,0)
  sampler.observe(text,{caret},100)
  assert.equal(sampler.decision('refine',{now:60000}).ready,false)
  sampler.observe(text,{caret:text.indexOf(' and')+1},60100)
  assert.equal(sampler.decision('refine',{now:63600}).ready,true)
})

test('pasted paragraphs can end without punctuation; a pasted single word cannot',()=>{
  assert.equal(contentBoundary('Develop a clear plan for this research','',{committed:true}).ready,true)
  assert.equal(contentBoundary('Python','',{committed:true}).ready,false)
  assert.equal(contentBoundary('Develop a clear plan for this resear').ready,false)
})

test('edits reset quiet time; each consumer samples once and uses its own cadence',()=>{
  const sampler=new ContentSampler('Use Python.',0)
  sampler.observe('Use Rust.',{},1800)
  assert.equal(sampler.decision('refine',{now:2000}).ready,false)
  assert.equal(sampler.decision('refine',{now:3800}).ready,true)
  sampler.mark('refine','',3800)
  assert.equal(sampler.decision('refine',{now:90000}).ready,false)
  assert.equal(sampler.decision('comments',{now:9999,interval:10000}).ready,false)
  assert.equal(sampler.decision('comments',{now:10000,interval:10000}).ready,true)
  sampler.mark('comments','',10000)
  assert.equal(sampler.decision('research',{now:10000,interval:10000}).ready,true)
  sampler.observe('Use Rust. Include tests.',{},11000)
  assert.equal(sampler.decision('comments',{now:13000,interval:10000}).ready,false)
  assert.equal(sampler.decision('comments',{now:20000,interval:10000}).ready,true)
})

test('whitespace edits do not request another generation, but a new model context can',()=>{
  const sampler=new ContentSampler('Use Python.',0)
  sampler.mark('refine','dream',2000)
  sampler.observe('Use  Python. ',{},2500)
  assert.equal(sampler.decision('refine',{now:20000,context:'dream'}).ready,false)
  assert.equal(sampler.decision('refine',{now:20000,context:'new-model'}).ready,true)
})

test('hardware latency can lengthen quiet time without bypassing content boundaries',()=>{
  const sampler=new ContentSampler('Use Python.',0)
  assert.equal(sampler.decision('refine',{now:5000,latency:8000}).ready,false)
  assert.equal(sampler.decision('refine',{now:8000,latency:8000}).ready,true)
  sampler.observe('Use Pyth',{},9000)
  assert.equal(sampler.decision('refine',{now:60000,latency:8000}).ready,false)
})
