import test from 'node:test'
import assert from 'node:assert/strict'
import { DraftBuffer, safeUrl } from '../src/client.js'

test('typing during an autosave preserves newer text and serializes revisions', async () => {
  let release
  const calls = []
  const buffer = new DraftBuffer({ text:'main', version:1 }, async (text,version) => {
    calls.push([text,version])
    if(calls.length===1) await new Promise(resolve=>{release=resolve})
    return {text,version:version+1}
  })
  buffer.edit('first'); const saving=buffer.flush()
  buffer.edit('second'); release(); await saving
  assert.deepEqual(calls,[['first',1],['second',2]])
  assert.equal(buffer.snapshot().text,'second')
  assert.equal(buffer.snapshot().dirty,false)
  buffer.close()
})

test('conflict preserves local draft and prevents automatic overwrite', async () => {
  let calls=0
  const buffer=new DraftBuffer({text:'original',version:1},async()=>{calls++;throw Object.assign(new Error('conflict'),{status:409})})
  buffer.edit('local edit')
  await assert.rejects(buffer.flush())
  await assert.rejects(buffer.flush())
  assert.equal(calls,1);assert.equal(buffer.text,'local edit');assert.equal(buffer.snapshot().dirty,true)
  buffer.close()
})

test('failed save keeps browser recovery and can be retried', async () => {
  const persisted=[];let fail=true
  const buffer=new DraftBuffer({text:'a',version:1},async(text,version)=>{if(fail)throw new Error('offline');return{text,version:version+1}},()=>{},value=>persisted.push(value))
  buffer.edit('b');await assert.rejects(buffer.flush())
  assert.equal(persisted.at(-1).text,'b')
  fail=false;await buffer.flush();assert.equal(persisted.at(-1),null);buffer.close()
})

test('source links reject script and local-file URL schemes',()=>{
  assert.equal(safeUrl('javascript:alert(1)'),null)
  assert.equal(safeUrl('file:///etc/passwd'),null)
  assert.equal(safeUrl('https://example.org/paper'),'https://example.org/paper')
})
