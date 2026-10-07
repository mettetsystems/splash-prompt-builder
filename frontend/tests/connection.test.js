import test from 'node:test'
import assert from 'node:assert/strict'
import { SplashConnection, websocketUrl } from '../src/connection.js'

class FakeSocket {
  static instances = []
  constructor(url) { this.url = url; this.readyState = 0; this.sent = []; FakeSocket.instances.push(this) }
  open() { this.readyState = 1; this.onopen?.() }
  send(message) { this.sent.push(JSON.parse(message)) }
  receive(message) { this.onmessage?.({ data: JSON.stringify(message) }) }
  close() { this.readyState = 3; this.onclose?.() }
}
const draft = (text, commit = false) => ({ text, cursor: text.length, freeze_radius: 3, commit })
const response = (request, output = request.text) => ({
  type: 'diffusion_update', request_id: request.request_id, source: request.text, output,
  segments: [{ text: request.text, kind: 'anchor' }, ...(output !== request.text ? [{ text: output.slice(request.text.length), kind: 'generated' }] : [])],
})
function setup(t) {
  t.mock.timers.enable({ apis: ['setTimeout'] })
  let state = {}
  const client = new SplashConnection('ws://localhost/ws', (patch) => { state = { ...state, ...patch } }, FakeSocket)
  client.start()
  const socket = client.socket
  socket.open()
  t.after(() => client.stop())
  return { client, socket, state: () => state, tick: (ms) => t.mock.timers.tick(ms) }
}

test('WebSocket uses current host and HTTPS protocol', () => {
  assert.equal(websocketUrl({ protocol: 'https:', host: 'splash.test:8443' }), 'wss://splash.test:8443/ws')
  assert.equal(websocketUrl({ protocol: 'http:', host: '127.0.0.1:5173' }), 'ws://127.0.0.1:5173/ws')
})
test('typing coalesces, renders after 100ms, and rejects stale updates', (t) => {
  const { client, socket, state, tick } = setup(t)
  client.update(draft('a')); tick(40)
  const old = socket.sent.at(-1)
  client.update(draft('ab')); client.update(draft('abc')); tick(40)
  assert.equal(socket.sent.at(-1).text, 'abc')
  socket.receive(response(old)); tick(100)
  assert.equal(state().result, undefined)
  socket.receive(response(socket.sent.at(-1), 'abc more'))
  tick(99); assert.equal(state().pending, true)
  tick(1); assert.equal(state().result.output, 'abc more')
  assert.equal(state().pending, false)
})
test('typing invalidates a response already queued for rendering', (t) => {
  const { client, socket, state, tick } = setup(t)
  client.update(draft('first')); tick(40)
  socket.receive(response(socket.sent.at(-1)))
  client.update(draft('second')); tick(100)
  assert.equal(state().result, undefined)
})
test('commit and clear immediately remove generated ownership', (t) => {
  const { client, state } = setup(t)
  client.update(draft('all mine', true))
  assert.deepEqual(state().result.segments, [{ kind: 'anchor', text: 'all mine' }])
  client.update(draft(''))
  assert.equal(state().result.output, '')
})
test('reconnect sends latest draft and cleanup cancels reconnects', (t) => {
  const { client, socket, state, tick } = setup(t)
  socket.close()
  assert.equal(state().connection, 'reconnecting')
  client.update(draft('while offline'))
  tick(500)
  const replacement = client.socket
  assert.notEqual(replacement, socket)
  replacement.open()
  assert.equal(replacement.sent.at(-1).text, 'while offline')
  replacement.close()
  client.stop(); tick(9000)
  assert.equal(client.socket, replacement)
})
test('composition pauses outgoing requests and then resumes', (t) => {
  const { client, socket, tick } = setup(t)
  const initial = socket.sent.length
  client.update(draft('世'), true); tick(40)
  assert.equal(socket.sent.length, initial)
  client.update(draft('世界'), false); tick(40)
  assert.equal(socket.sent.at(-1).text, '世界')
})
test('invalid anchor output is ignored and timeouts expose an error', (t) => {
  const { client, socket, state, tick } = setup(t)
  client.update(draft('sacred')); tick(40)
  const message = response(socket.sent.at(-1))
  message.segments = [{ text: 'changed', kind: 'anchor' }]
  message.output = 'changed'
  socket.receive(message); tick(8000)
  assert.equal(state().result, undefined)
  assert.match(state().error, /timed out/)
})
test('server validation errors stop the pending state', (t) => {
  const { client, socket, state, tick } = setup(t)
  client.update(draft('x')); tick(40)
  socket.receive({ type: 'error', request_id: socket.sent.at(-1).request_id, message: 'Invalid cursor' })
  assert.equal(state().pending, false)
  assert.equal(state().error, 'Invalid cursor')
})
