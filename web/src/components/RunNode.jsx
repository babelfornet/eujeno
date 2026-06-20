import { useState, useRef, useEffect } from 'react'
import { container, sectionLabel, h2, mono } from '../styles.js'

const INSTALL_CMD = 'curl -fsSL https://get.eujeno.net | sh'

const CHECKS = [
  { title: 'Any modern GPU', body: '8 GB VRAM is enough to host a few layers.' },
  { title: 'A public port', body: 'So neighbours can pass activations to you.' },
  { title: "That's it", body: 'No registration. The node auto-joins on boot.' },
]

export default function RunNode() {
  const [copied, setCopied] = useState(false)
  const timer = useRef(null)

  useEffect(() => () => clearTimeout(timer.current), [])

  const copy = () => {
    try {
      navigator.clipboard && navigator.clipboard.writeText(INSTALL_CMD)
    } catch (e) {
      /* clipboard unavailable */
    }
    setCopied(true)
    clearTimeout(timer.current)
    timer.current = setTimeout(() => setCopied(false), 1500)
  }

  return (
    <section id="run" style={{ ...container, padding: 'var(--section-pad) 28px' }}>
      <div
        className="run-grid"
        style={{ display: 'grid', gridTemplateColumns: '0.9fr 1.1fr', gap: '48px', alignItems: 'center' }}
      >
        <div>
          <div style={sectionLabel}>Join the swarm</div>
          <h2 style={h2}>Run a node in one command.</h2>
          <p style={{ margin: '18px 0 0', fontSize: '16px', lineHeight: 1.6, color: 'var(--muted)', maxWidth: '420px' }}>
            Point it at the network, let it claim the layers your GPU can hold, and start serving. The swarm handles
            discovery, routing, and failover.
          </p>
          <div style={{ marginTop: '28px', display: 'flex', flexDirection: 'column', gap: '14px' }}>
            {CHECKS.map((c) => (
              <div key={c.title} style={{ display: 'flex', gap: '13px', alignItems: 'flex-start' }}>
                <span
                  style={{
                    width: '22px',
                    height: '22px',
                    borderRadius: '50%',
                    background: 'color-mix(in srgb, var(--accent) 15%, transparent)',
                    color: 'var(--accent)',
                    display: 'grid',
                    placeItems: 'center',
                    fontSize: '12px',
                    fontWeight: 700,
                    flex: 'none',
                  }}
                >
                  ✓
                </span>
                <div>
                  <div style={{ fontSize: '15.5px', color: 'var(--text)', fontWeight: 600 }}>{c.title}</div>
                  <div style={{ fontSize: '14px', color: 'var(--muted2)' }}>{c.body}</div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* code card (slate, fixed dark) */}
        <div
          style={{
            border: '1px solid #1c2230',
            background: '#0f1320',
            borderRadius: 'var(--radius)',
            overflow: 'hidden',
            boxShadow: 'var(--float-shadow)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '13px 16px', borderBottom: '1px solid #1c2230' }}>
            <span style={{ width: '10px', height: '10px', borderRadius: '50%', background: '#2a3344' }} />
            <span style={{ width: '10px', height: '10px', borderRadius: '50%', background: '#2a3344' }} />
            <span style={{ width: '10px', height: '10px', borderRadius: '50%', background: '#2a3344' }} />
            <span style={{ marginLeft: '6px', fontFamily: mono, fontSize: '11.5px', color: '#6b7689' }}>eujeno — node</span>
            <button
              type="button"
              className="copy-btn"
              onClick={copy}
              style={{
                marginLeft: 'auto',
                fontFamily: mono,
                fontSize: '11px',
                color: '#aab4c5',
                background: 'transparent',
                border: '1px solid #2a3344',
                borderRadius: '7px',
                padding: '4px 10px',
                cursor: 'pointer',
              }}
            >
              {copied ? 'copied ✓' : 'copy'}
            </button>
          </div>
          <div style={{ padding: '20px 18px', fontFamily: mono, fontSize: '13px', lineHeight: 1.9, color: '#dce3ee' }}>
            <div>
              <span style={{ color: '#8b86ff' }}>$</span> curl -fsSL https://get.eujeno.net | sh
            </div>
            <div>
              <span style={{ color: '#8b86ff' }}>$</span> eujeno up --model swarm/llama-70b --layers auto
            </div>
            <div style={{ height: '10px' }} />
            <div style={{ color: '#6b7689' }}>
              {'  '}resolving peers ..... <span style={{ color: '#6ee7a8' }}>142 found</span>
            </div>
            <div style={{ color: '#6b7689' }}>
              {'  '}claiming layers ..... <span style={{ color: '#dce3ee' }}>L24–L31</span>
            </div>
            <div style={{ color: '#6b7689' }}>
              {'  '}loading weights ..... <span style={{ color: '#6ee7a8' }}>ok</span>
            </div>
            <div style={{ color: '#6b7689' }}>
              {'  '}joining chain ....... <span style={{ color: '#6ee7a8' }}>ok</span>
            </div>
            <div style={{ height: '10px' }} />
            <div>
              <span style={{ color: '#6ee7a8' }}>●</span> serving <span style={{ color: '#6b7689' }}>· 0 errors · 38 tok/s out</span>{' '}
              <span
                style={{
                  display: 'inline-block',
                  width: '7px',
                  height: '14px',
                  background: '#8b86ff',
                  verticalAlign: 'middle',
                  animation: 'blink 1s steps(2) infinite',
                }}
              />
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
