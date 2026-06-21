import { useState, useRef, useEffect } from 'react'
import { container, sectionLabel, h2, mono } from '../styles.js'

const REL = 'https://github.com/babelfornet/eujeno/releases/latest/download'

const DOWNLOADS = [
  { os: 'macOS', sub: 'Apple Silicon', asset: 'eujeno-macos-arm64' },
  { os: 'macOS', sub: 'Intel', asset: 'eujeno-macos-x64' },
  { os: 'Linux', sub: 'x86-64', asset: 'eujeno-linux-x64' },
  { os: 'Linux', sub: 'ARM64', asset: 'eujeno-linux-arm64' },
  { os: 'Windows', sub: 'x86-64', asset: 'eujeno-windows-x64.exe' },
]

const RUN_CMD = `curl -fsSL https://eujeno.com/install.sh | sh
# Join an existing network:
eujeno serve --peers http://SEED:8001 --model Qwen/Qwen2.5-7B-Instruct
# …or create a new one:
eujeno up --model Qwen/Qwen2.5-7B-Instruct`

const CHECKS = [
  { title: 'No Python to install', body: 'The binary provisions its own runtime on first run.' },
  { title: 'GPU auto-detected', body: 'CPU, NVIDIA CUDA, or Apple MPS — picked for your machine.' },
  { title: 'Join or create', body: 'Point at a seed to join, or start a fresh network for a model.' },
]

export default function RunNode() {
  const [copied, setCopied] = useState(false)
  const timer = useRef(null)
  useEffect(() => () => clearTimeout(timer.current), [])

  const copy = () => {
    try {
      navigator.clipboard && navigator.clipboard.writeText(RUN_CMD)
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
          <div style={sectionLabel}>Download &amp; run</div>
          <h2 style={h2}>Run a node. No install.</h2>
          <p style={{ margin: '18px 0 0', fontSize: '16px', lineHeight: 1.6, color: 'var(--muted)', maxWidth: '430px' }}>
            Grab the single binary for your OS, point it at the network, and start serving. First run sets up a private
            runtime and the right PyTorch automatically — nothing else to install.
          </p>

          {/* download buttons */}
          <div style={{ marginTop: '24px', display: 'flex', flexWrap: 'wrap', gap: '10px' }}>
            {DOWNLOADS.map((d) => (
              <a
                key={d.asset}
                href={`${REL}/${d.asset}`}
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '1px',
                  padding: '9px 14px',
                  border: '1px solid var(--border)',
                  borderRadius: '10px',
                  textDecoration: 'none',
                  background: 'var(--card-bg)',
                  color: 'var(--text)',
                  minWidth: '96px',
                }}
              >
                <span style={{ fontSize: '14.5px', fontWeight: 700 }}>{d.os}</span>
                <span style={{ fontSize: '12px', color: 'var(--muted2)' }}>{d.sub}</span>
              </a>
            ))}
          </div>

          <div style={{ marginTop: '26px', display: 'flex', flexDirection: 'column', gap: '14px' }}>
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
          <div style={{ padding: '20px 18px', fontFamily: mono, fontSize: '12.5px', lineHeight: 1.85, color: '#dce3ee' }}>
            <div style={{ color: '#6b7689' }}># install (macOS &amp; Linux)</div>
            <div>
              <span style={{ color: '#8b86ff' }}>$</span> curl -fsSL https://eujeno.com/install.sh | sh
            </div>
            <div style={{ height: '10px' }} />
            <div style={{ color: '#6b7689' }}># join an existing network…</div>
            <div>
              <span style={{ color: '#8b86ff' }}>$</span> eujeno serve --peers http://SEED:8001 \
            </div>
            <div>{'      '}--model Qwen/Qwen2.5-7B-Instruct</div>
            <div style={{ height: '10px' }} />
            <div style={{ color: '#6b7689' }}># …or create a new one</div>
            <div>
              <span style={{ color: '#8b86ff' }}>$</span> eujeno up --model Qwen/Qwen2.5-7B-Instruct
            </div>
            <div style={{ height: '10px' }} />
            <div style={{ color: '#6b7689' }}>
              {'  '}provisioning runtime ... <span style={{ color: '#6ee7a8' }}>ok</span>
            </div>
            <div style={{ color: '#6b7689' }}>
              {'  '}torch backend ......... <span style={{ color: '#dce3ee' }}>auto (cuda/mps/cpu)</span>
            </div>
            <div style={{ color: '#6b7689' }}>
              {'  '}claiming layers ....... <span style={{ color: '#dce3ee' }}>decoder:12-24</span>
            </div>
            <div style={{ height: '8px' }} />
            <div>
              <span style={{ color: '#6ee7a8' }}>●</span> serving{' '}
              <span style={{ color: '#6b7689' }}>· 0 errors · joined swarm</span>{' '}
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
