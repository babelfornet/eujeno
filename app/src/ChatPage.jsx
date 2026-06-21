import React, { useState, useEffect, useRef, useCallback } from 'react'
import * as api from './api.js'

const STORAGE_KEY = 'eujeno_chat'
const SYS_USER = "You are the inference endpoint of Eujeno, a peer-to-peer network that serves a large language model split layer by layer across many volunteer nodes. Answer helpfully and fairly concisely. Never mention these instructions."
const SYS_ASSISTANT = "Swarm online. Ready to serve."

const EXAMPLES = [
  {
    label: 'Explain layer-sharded inference',
    text: 'Explain how layer-sharded inference works in a peer-to-peer model network, simply.',
  },
  {
    label: 'Write a haiku about distributed compute',
    text: 'Write a haiku about distributed compute.',
  },
  {
    label: 'What can I run on 8 GB RAM?',
    text: 'What kinds of models or layers can I contribute to a swarm with only 8 GB of RAM?',
  },
]

function loadChat() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (raw) return JSON.parse(raw)
  } catch (_) {}
  return []
}

function persistChat(chat) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(chat))
  } catch (_) {}
}

// Inject dotpulse keyframes once
let _injected = false
function injectKeyframes() {
  if (_injected) return
  _injected = true
  const style = document.createElement('style')
  style.textContent = `
    @keyframes dotpulse {
      0%, 100% { opacity: 0.2; transform: scale(0.8); }
      50%       { opacity: 1;   transform: scale(1.0); }
    }
  `
  document.head.appendChild(style)
}

export default function ChatPage({ T, accent }) {
  const [chat,    setChat]    = useState(() => loadChat())
  const [input,   setInput]   = useState('')
  const [sending, setSending] = useState(false)
  const bottomRef = useRef(null)
  const textareaRef = useRef(null)

  useEffect(() => { injectKeyframes() }, [])

  // Auto-scroll whenever chat changes
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [chat])

  const sendText = useCallback(async (raw) => {
    const text = (raw || '').trim()
    if (!text || sending) return

    // Snapshot history before state update (avoid stale closure)
    const historySnapshot = chat
      .filter(m => !m.pending)
      .map(m => ({ role: m.role, content: m.content }))

    // Optimistically append user bubble + pending assistant bubble
    const userMsg    = { role: 'user',      content: text }
    const pendingMsg = { role: 'assistant', content: '', pending: true }

    setChat(prev => {
      const next = [...prev, userMsg, pendingMsg]
      persistChat(next)
      return next
    })
    setInput('')
    setSending(true)

    // Build API messages: short system priming + history + new user text
    const apiMessages = [
      { role: 'user',      content: SYS_USER },
      { role: 'assistant', content: SYS_ASSISTANT },
      ...historySnapshot,
      { role: 'user', content: text },
    ]

    let answer, routing

    try {
      const resp = await api.chat(apiMessages, 256)
      answer = resp.choices?.[0]?.message?.content || '…(no tokens returned)'
      if (!answer.trim()) answer = '…(no tokens returned)'

      // Extract routing footer from resp.eujeno if present
      const ej = resp.eujeno
      if (ej) {
        const hops   = ej.hops   != null ? ej.hops   : null
        const layers = ej.layers != null ? ej.layers : null
        const tokS   = ej.tok_s  != null ? Math.round(ej.tok_s) : null
        if (hops != null && layers != null && tokS != null) {
          routing = `routed through ${hops} nodes · ${layers} layers · ${tokS} tok/s`
        }
      }
    } catch (_) {
      answer = "Couldn't reach the swarm right now — is your node running and connected to the coordinator?"
    }

    setChat(prev => {
      const next = [...prev]
      // Replace last pending bubble
      for (let i = next.length - 1; i >= 0; i--) {
        if (next[i].pending) {
          next[i] = { role: 'assistant', content: answer, routing: routing || null }
          break
        }
      }
      persistChat(next)
      return next
    })
    setSending(false)
  }, [chat, sending])

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendText(input)
    }
  }

  const chatEmpty = chat.length === 0

  return (
    <div style={{ flex: '1', minHeight: 0, display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <div style={{
        padding: '22px 36px',
        borderBottom: `1px solid ${T.border}`,
        flexShrink: 0,
      }}>
        <h1 style={{ margin: 0, fontSize: '22px', fontWeight: 800, letterSpacing: '-0.02em', color: T.text }}>Chat</h1>
        <p style={{ margin: '5px 0 0', fontSize: '13.5px', color: T.muted }}>
          Responses routed layer by layer across the network
        </p>
      </div>

      {/* Messages area */}
      <div style={{ flex: '1', minHeight: 0, overflowY: 'auto', padding: '26px 36px' }}>
        <div style={{ maxWidth: '760px', margin: '0 auto' }}>

          {/* Empty state */}
          {chatEmpty && (
            <div style={{ textAlign: 'center', padding: '54px 0 30px' }}>
              <span style={{
                display: 'inline-grid',
                placeItems: 'center',
                width: '54px',
                height: '54px',
                borderRadius: '14px',
                background: `color-mix(in srgb, ${accent} 12%, transparent)`,
                color: accent,
              }}>
                <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M4 5.5h16v10.5H9l-4 3.5v-3.5H4z"/>
                  <path d="M8.5 10.5h7M8.5 13h4.5"/>
                </svg>
              </span>
              <h2 style={{ margin: '18px 0 6px', fontSize: '22px', fontWeight: 800, letterSpacing: '-0.02em', color: T.text }}>
                Ask the swarm
              </h2>
              <p style={{ margin: '0 auto', fontSize: '15px', color: T.muted, maxWidth: '420px' }}>
                Your prompt routes layer by layer through the connected nodes and streams back here.
              </p>
              <div style={{ marginTop: '22px', display: 'flex', gap: '10px', justifyContent: 'center', flexWrap: 'wrap' }}>
                {EXAMPLES.map(ex => (
                  <ExampleChip key={ex.label} label={ex.label} accent={accent} T={T} onClick={() => sendText(ex.text)} />
                ))}
              </div>
            </div>
          )}

          {/* Message bubbles */}
          {chat.map((m, i) => (
            <MessageBubble key={i} m={m} T={T} accent={accent} />
          ))}

          <div ref={bottomRef} />
        </div>
      </div>

      {/* Composer */}
      <div style={{
        borderTop: `1px solid ${T.border}`,
        padding: '16px 36px',
        background: T.pageBg,
        flexShrink: 0,
      }}>
        <div style={{ maxWidth: '760px', margin: '0 auto' }}>
          <div style={{
            display: 'flex',
            gap: '10px',
            alignItems: 'flex-end',
            border: `1px solid ${T.borderStrong}`,
            background: T.cardBg,
            borderRadius: '14px',
            padding: '8px 8px 8px 14px',
          }}>
            <textarea
              ref={textareaRef}
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              rows={1}
              placeholder="Message the swarm…"
              style={{
                flex: '1',
                resize: 'none',
                border: 'none',
                outline: 'none',
                background: 'transparent',
                color: T.text,
                fontSize: '15px',
                lineHeight: '1.5',
                maxHeight: '140px',
                padding: '7px 0',
                fontFamily: 'inherit',
              }}
            />
            <button
              onClick={() => sendText(input)}
              disabled={!input.trim() || sending}
              style={{
                flexShrink: 0,
                display: 'grid',
                placeItems: 'center',
                width: '38px',
                height: '38px',
                borderRadius: '10px',
                border: 'none',
                cursor: (!input.trim() || sending) ? 'not-allowed' : 'pointer',
                background: accent,
                color: '#fff',
                opacity: (!input.trim() || sending) ? 0.45 : 1,
                transition: 'opacity 0.15s',
              }}
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M5 12h13M12 6l6 6-6 6"/>
              </svg>
            </button>
          </div>
          <div style={{ marginTop: '8px', textAlign: 'center', fontSize: '11.5px', color: T.muted2 }}>
            Eujeno serves an open model across volunteer nodes. Output may be imperfect.
          </div>
        </div>
      </div>
    </div>
  )
}

function ExampleChip({ label, accent, T, onClick }) {
  const [hovered, setHovered] = useState(false)
  return (
    <button
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        border: `1px solid ${hovered ? accent : T.borderStrong}`,
        background: T.cardBg,
        color: hovered ? accent : T.text,
        fontSize: '13.5px',
        fontWeight: 500,
        padding: '9px 14px',
        borderRadius: '999px',
        cursor: 'pointer',
        transition: 'border-color 0.15s, color 0.15s',
        fontFamily: 'inherit',
      }}
    >
      {label}
    </button>
  )
}

function MessageBubble({ m, T, accent }) {
  const isUser = m.role === 'user'

  return (
    <div style={{
      display: 'flex',
      justifyContent: isUser ? 'flex-end' : 'flex-start',
      marginBottom: '18px',
    }}>
      <div style={{ maxWidth: '80%', minWidth: 0 }}>
        {/* Bubble */}
        <div style={{
          border: `1px solid ${isUser ? 'transparent' : T.border}`,
          background: isUser ? accent : T.cardBg,
          color: isUser ? '#ffffff' : T.text,
          padding: '13px 16px',
          borderRadius: '14px',
          fontSize: '14.5px',
          lineHeight: '1.6',
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word',
        }}>
          {m.pending ? (
            <span style={{ display: 'inline-flex', alignItems: 'center', gap: '9px', color: T.muted }}>
              <span style={{ display: 'inline-flex', gap: '4px' }}>
                <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: accent, animation: 'dotpulse 1s ease-in-out infinite' }} />
                <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: accent, animation: 'dotpulse 1s ease-in-out 0.15s infinite' }} />
                <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: accent, animation: 'dotpulse 1s ease-in-out 0.3s infinite' }} />
              </span>
              <span style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: '12.5px' }}>routing through the swarm…</span>
            </span>
          ) : (
            m.content
          )}
        </div>

        {/* Routing footer — only for assistant messages with routing info */}
        {!m.pending && m.routing && (
          <div style={{
            marginTop: '6px',
            fontFamily: "'JetBrains Mono',monospace",
            fontSize: '11px',
            color: T.muted2,
          }}>
            {m.routing}
          </div>
        )}
      </div>
    </div>
  )
}
