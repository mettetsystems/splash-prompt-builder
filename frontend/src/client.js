export async function api(path, { method = 'GET', body, ...options } = {}) {
  const response = await fetch(`/api${path}`, {
    method, ...options,
    headers: body instanceof FormData ? undefined : { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : body instanceof FormData ? body : JSON.stringify(body),
  })
  const data = await response.json()
  if (!response.ok) {
    const error = new Error(typeof data.detail === 'string' ? data.detail : 'Please check the supplied fields.')
    error.status = response.status
    throw error
  }
  return data
}

// Serialize autosaves without blocking typing or overwriting newer local edits.
export class DraftBuffer {
  constructor(draft, save, notify = () => {}, persist = () => {}) {
    this.text = this.saved = draft.text
    this.version = draft.version
    Object.assign(this, { save, notify, persist, pending: null, error: null, timer: null, closed: false })
  }
  snapshot() { return { text: this.text, version: this.version, dirty: this.text !== this.saved, saving: !!this.pending, error: this.error } }
  emit() { if (!this.closed) this.notify(this.snapshot()) }
  edit(text) {
    this.text = text
    this.persist({ text, version: this.version }); this.emit()
    clearTimeout(this.timer)
    this.timer = setTimeout(() => this.flush().catch(() => {}), 450)
  }
  async flush() {
    clearTimeout(this.timer)
    if (this.pending) { await this.pending; return this.flush() }
    if (this.error?.status === 409) throw this.error
    while (!this.closed && this.text !== this.saved) {
      const text = this.text
      this.error = null
      this.pending = this.save(text, this.version); this.emit()
      try {
        const draft = await this.pending
        this.saved = text; this.version = draft.version
        this.persist(this.text === this.saved ? null : { text: this.text, version: this.version })
      } catch (error) { this.error = error; throw error }
      finally { this.pending = null; this.emit() }
    }
    return this.version
  }
  replace(draft) {
    this.text = this.saved = draft.text; this.version = draft.version; this.error = null
    this.persist(null); this.emit()
  }
  close() { this.closed = true; clearTimeout(this.timer) }
}

export function safeUrl(value) {
  try { const url = new URL(value); return ['https:', 'http:'].includes(url.protocol) ? url.href : null } catch { return null }
}

export function isEnrichment(proposal) {
  return proposal.rationale === 'Language and instruction precision'
}

// Backend offsets count Unicode code points, not JavaScript UTF-16 units.
// Keep research additions out of the live rewrite until individually approved.
export function liveEnrichment(draft, proposals) {
  const edits = proposals.filter(isEnrichment)
  const original = Array.from(draft.text)
  let result = [...original], boundary = original.length + 1
  for (const edit of [...edits].sort((a, b) => b.start - a.start)) {
    if (edit.version !== draft.version || !Number.isInteger(edit.start) || !Number.isInteger(edit.end) ||
        edit.start < 0 || edit.end < edit.start || edit.end > original.length || edit.end > boundary ||
        original.slice(edit.start, edit.end).join('') !== edit.before) {
      return {text: draft.text, ids: [], version: draft.version}
    }
    result = [...result.slice(0, edit.start), ...Array.from(edit.after), ...result.slice(edit.end)]
    boundary = edit.start
  }
  return {text: result.join(''), ids: edits.map(edit => edit.id), version: draft.version}
}
