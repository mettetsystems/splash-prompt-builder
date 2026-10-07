import React, { useEffect, useRef, useState } from 'react'
import useSplash from '../hooks/useSplash.js'
import { MAX_PROMPT_LENGTH } from '../connection.js'
import './Editor.css'

const EXAMPLES = [
  ['A quiet place', 'a quiet cabin beside a lake at sunrise'],
  ['Something otherworldly', 'an astronaut tending a garden on the moon'],
  ['A small story', 'a fox exploring an old bookshop on a rainy evening'],
]

export default function Editor() {
  const [draft, setDraft] = useState({ text: '', cursor: 0, freeze_radius: 3, commit: false })
  const [composing, setComposing] = useState(false)
  const [notice, setNotice] = useState('')
  const textarea = useRef(null)
  const noticeTimer = useRef(null)
  const { connection, pending, error, result } = useSplash(draft, composing)
  const connected = connection === 'connected'
  const fresh = result?.source === draft.text && !pending && !error && !composing
  const hasAdditions = result?.segments.some((part) => part.kind === 'generated')
  const tooLong = (result?.output.length || 0) > MAX_PROMPT_LENGTH
  const canCommit = connected && fresh && hasAdditions && !tooLong
  const canCopy = fresh && Boolean(result?.output)
  const wordCount = draft.text.trim() ? draft.text.trim().split(/\s+/u).length : 0
  const frozenCount = result?.segments.filter((part) => part.kind === 'generated' && part.frozen).length || 0

  useEffect(() => () => clearTimeout(noticeTimer.current), [])

  function announce(message) {
    clearTimeout(noticeTimer.current)
    setNotice(message)
    noticeTimer.current = setTimeout(() => setNotice(''), 3000)
  }

  function replaceText(text, commit = false) {
    setNotice('')
    setDraft((previous) => ({ ...previous, text, cursor: text.length, commit }))
    textarea.current?.focus()
    requestAnimationFrame(() => textarea.current?.setSelectionRange(text.length, text.length))
  }

  function commit() {
    if (!canCommit) return
    replaceText(result.output, true)
    announce('Preview committed. Every word is now yours.')
  }

  async function copy() {
    if (!canCopy) return
    try {
      await navigator.clipboard.writeText(result.output)
      announce('Prompt copied to clipboard.')
    } catch {
      announce('Clipboard unavailable. Select the preview text to copy it.')
    }
  }

  function select(event) {
    const cursor = event.currentTarget.selectionStart
    setDraft((previous) => previous.cursor === cursor ? previous : { ...previous, cursor })
  }

  return (
    <section className="workspace" aria-label="Prompt playground">
      <div className="workspace-toolbar">
        <div className="workspace-label"><span className="mini-mark" aria-hidden="true">✳</span> Prompt studio <span className="demo-badge">MOCK DEMO</span></div>
        <div className={`connection-status ${connected ? 'connected' : ''}`} role="status">
          <span className="status-dot" />
          {connected ? 'Connected' : connection === 'connecting' ? 'Connecting…' : 'Reconnecting…'}
        </div>
      </div>

      {!connected && <div className="connection-notice">Waiting for the local backend. Your draft stays here; Splash reconnects automatically.</div>}
      {error && <div className="error-notice" role="alert">{error}</div>}

      <div className="split-pane">
        <section className="editor-pane input-pane" aria-labelledby="input-title">
          <div className="pane-heading"><div><span className="step-number">01</span><h2 id="input-title"><label htmlFor="prompt-input">Your starting point</label></h2></div><span className="pane-tag">THE ANCHOR</span></div>
          <p className="pane-description" id="input-help">Your words stay exactly as you write them.</p>
          <textarea
            id="prompt-input" ref={textarea} value={draft.text} maxLength={MAX_PROMPT_LENGTH}
            onChange={(event) => {
              const { value, selectionStart } = event.currentTarget
              setDraft((previous) => ({ ...previous, text: value, cursor: selectionStart, commit: false }))
              setNotice('')
            }}
            onSelect={select}
            onCompositionStart={() => setComposing(true)}
            onCompositionEnd={() => setComposing(false)}
            onKeyDown={(event) => {
              if (event.key === 'Tab' && !event.shiftKey && !event.altKey && !event.ctrlKey && !event.metaKey && !event.nativeEvent.isComposing && canCommit) {
                event.preventDefault()
                commit()
              }
            }}
            placeholder={'A place, a feeling, a half-formed idea…\n\nStart typing and see where it goes.'}
            aria-describedby="input-help keyboard-help" spellCheck="true" className="input-area"
          />
          <div className="pane-bottom"><span>{wordCount} {wordCount === 1 ? 'word' : 'words'} <span className="divider">/</span> {draft.text.length.toLocaleString()} / 8,000 characters</span><button type="button" className="text-button" disabled={!draft.text} onClick={() => replaceText('')}>Clear</button></div>
        </section>

        <section className="editor-pane output-pane" aria-labelledby="output-title" aria-busy={pending && connected}>
          <div className="pane-heading"><div><span className="step-number">02</span><h2 id="output-title">A little more possibility</h2></div><span className={`live-label ${pending && connected ? 'updating' : ''}`}>{pending && connected ? 'Updating…' : 'LIVE PREVIEW'}</span></div>
          <p className="pane-description">New details find their place around your idea.</p>
          <div className="output-area" aria-label="Enriched prompt" tabIndex={0}>
            {result?.output ? result.segments.map((part, index) => (
              <span key={index} className={`${part.kind}-token${part.frozen ? ' frozen-token' : ''}`} title={part.kind === 'generated' ? (part.frozen ? 'Generated · frozen outside the cursor radius' : 'Generated · near the cursor') : undefined}>{part.text}</span>
            )) : <div className="empty-state"><span className="empty-icon" aria-hidden="true">✳</span><h3>Good things start small.</h3><p>Drop an idea on the left.<br />We’ll make a little room for more.</p></div>}
          </div>
          <div className="pane-bottom preview-bottom"><span className="legend"><span><i className="anchor-swatch" /> Your words</span><span><i className="generated-swatch" /> Added details</span></span><button type="button" className="text-button copy-button" onClick={copy} disabled={!canCopy}>Copy prompt <span aria-hidden="true">↗</span></button></div>
        </section>
      </div>

      <div className="workspace-controls">
        <div className="freeze-control"><div><label htmlFor="freeze-radius">Keep distant details steady</label><span>{frozenCount} frozen</span></div><div className="radius-row"><input id="freeze-radius" type="range" min="0" max="12" value={draft.freeze_radius} onChange={(event) => setDraft((previous) => ({ ...previous, freeze_radius: Number(event.target.value) }))} aria-describedby="freeze-help" /><output htmlFor="freeze-radius">±{draft.freeze_radius} words</output></div><p id="freeze-help">Only details near your cursor can change.</p></div>
        <div className="commit-control"><button type="button" className="commit-button" onClick={commit} disabled={!canCommit}>Make it mine <span aria-hidden="true">↵</span></button><p id="keyboard-help">{tooLong ? 'Preview exceeds 8,000 characters. Shorten your input to commit.' : <>Commit the preview, or press <kbd>Tab</kbd>. <kbd>Shift</kbd> + <kbd>Tab</kbd> moves focus back.</>}</p></div>
      </div>
      <div className="announcement" role="status" aria-live="polite">{notice || (draft.commit && result?.source === draft.text ? 'All yours. Keep writing to explore more.' : 'A rule-based preview to explore the interaction. No language model is running.')}</div>

      <div className="examples"><span>NEED A SPARK?</span>{EXAMPLES.map(([label, text]) => <button type="button" key={label} onClick={() => replaceText(text)}>{label} <span aria-hidden="true">↗</span></button>)}</div>
    </section>
  )
}
