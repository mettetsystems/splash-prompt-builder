export const MAX_PROMPT_LENGTH = 8000

export function websocketUrl(location) {
  return `${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws`
}

// Kept independent of React so reconnects, stale responses and teardown can be
// exercised without a browser. A connection owns exactly one draft and socket.
export class SplashConnection {
  constructor(url, notify, Socket = WebSocket) {
    this.url = url
    this.notify = notify
    this.Socket = Socket
    this.requestId = 0
    this.draft = { text: '', cursor: 0, freeze_radius: 3, commit: false }
    this.retryDelay = 500
    this.stopped = false
    this.suspended = false
  }

  start() {
    if (this.stopped) return
    this.notify({ connection: 'connecting' })
    const socket = new this.Socket(this.url)
    this.socket = socket
    socket.onopen = () => {
      if (this.stopped) return
      this.retryDelay = 500
      this.notify({ connection: 'connected', error: null })
      this.send()
    }
    socket.onmessage = (event) => {
      if (this.stopped) return
      let message
      try { message = JSON.parse(event.data) } catch { return }
      if (!message || message.request_id !== this.requestId) return
      if (message.type === 'error') {
        clearTimeout(this.responseTimer)
        this.notify({ pending: false, error: message.message || 'Preview unavailable.' })
        return
      }
      if (message.type !== 'diffusion_update' || message.source !== this.draft.text ||
          typeof message.output !== 'string' || !Array.isArray(message.segments)) return
      const valid = message.segments.every((part) => part && typeof part.text === 'string' &&
        (part.kind === 'anchor' || part.kind === 'generated'))
      if (!valid || message.segments.map((part) => part.text).join('') !== message.output ||
          message.segments.filter((part) => part.kind === 'anchor').map((part) => part.text).join('') !== this.draft.text) return
      clearTimeout(this.responseTimer)
      clearTimeout(this.renderTimer)
      this.renderTimer = setTimeout(() => {
        if (!this.stopped && message.request_id === this.requestId) {
          this.notify({ result: message, pending: false, error: null })
        }
      }, 100)
    }
    socket.onerror = () => socket.close()
    socket.onclose = () => {
      if (this.stopped) return
      this.clearWork()
      this.notify({ connection: 'reconnecting', pending: false })
      this.retryTimer = setTimeout(() => this.start(), this.retryDelay)
      this.retryDelay = Math.min(this.retryDelay * 2, 8000)
    }
  }

  clearWork() {
    clearTimeout(this.sendTimer)
    clearTimeout(this.renderTimer)
    clearTimeout(this.responseTimer)
  }

  update(draft, suspended = false) {
    this.draft = { ...draft }
    this.suspended = suspended
    this.requestId += 1
    this.clearWork()
    const immediate = !draft.text || draft.commit
    this.notify({ pending: !immediate, error: null, ...(immediate ? {
      result: { source: draft.text, output: draft.text,
        segments: draft.text ? [{ kind: 'anchor', text: draft.text }] : [] },
    } : {}) })
    this.sendTimer = setTimeout(() => this.send(), 40)
  }

  send() {
    clearTimeout(this.sendTimer)
    if (this.stopped || this.suspended || this.socket?.readyState !== 1) return
    this.notify({ pending: Boolean(this.draft.text && !this.draft.commit) })
    try {
      this.socket.send(JSON.stringify({ type: 'prompt_update', request_id: this.requestId, ...this.draft }))
      clearTimeout(this.responseTimer)
      this.responseTimer = setTimeout(() => {
        this.notify({ pending: false, error: 'The preview timed out. Edit your prompt to try again.' })
      }, 8000)
    } catch {
      this.socket.close()
    }
  }

  stop() {
    this.stopped = true
    this.clearWork()
    clearTimeout(this.retryTimer)
    if (this.socket) {
      this.socket.onopen = this.socket.onmessage = this.socket.onclose = this.socket.onerror = null
      this.socket.close()
    }
  }
}
