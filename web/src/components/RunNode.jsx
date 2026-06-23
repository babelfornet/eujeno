import { useState, useRef, useEffect } from 'react'
import { container, sectionLabel, h2, mono } from '../styles.js'

const REL = 'https://github.com/babelfornet/eujeno/releases/latest/download'
const RELEASES = 'https://github.com/babelfornet/eujeno/releases/latest'

// One entry per OS tile. The terminal card is rendered from the selected
// entry: prompt symbol ($ vs PowerShell >), binary name, line-continuation
// char (\ vs `), the install one-liner, and the torch backend hint all switch.
const OSES = [
  {
    id: 'macOS',
    label: 'macOS',
    sub: 'Apple Silicon · Intel',
    prompt: '$',
    bin: 'eujeno',
    cont: '\\',
    installComment: '# install (macOS)',
    install: ['curl -fsSL https://eujeno.com/install.sh | sh'],
    torch: 'auto (mps/cpu)',
  },
  {
    id: 'Windows',
    label: 'Windows',
    sub: 'x86-64 · PowerShell',
    prompt: '>',
    bin: '.\\eujeno.exe',
    cont: '`',
    installComment: '# install (Windows · PowerShell)',
    install: [`irm ${REL}/eujeno-windows-x64.exe \``, '    -OutFile eujeno.exe'],
    torch: 'auto (cuda/cpu)',
  },
  {
    id: 'Linux',
    label: 'Linux',
    sub: 'x86-64 · ARM64',
    prompt: '$',
    bin: 'eujeno',
    cont: '\\',
    installComment: '# install (Linux)',
    install: ['curl -fsSL https://eujeno.com/install.sh | sh'],
    torch: 'auto (cuda/cpu)',
  },
]

const CHECKS = [
  { title: 'No Python to install', body: 'The binary provisions its own runtime on first run.' },
  { title: 'GPU auto-detected', body: 'CPU, NVIDIA CUDA, or Apple MPS — picked for your machine.' },
  { title: 'Join or create', body: 'Point at a seed to join, or start a fresh network for a model.' },
]

// The plain-text version of the selected OS's commands, for the copy button.
function copyText(os) {
  return [
    os.installComment,
    ...os.install,
    '',
    '# join an existing network',
    `${os.bin} serve --peers http://SEED:8001 ${os.cont}`,
    '    --model Qwen/Qwen2.5-7B-Instruct',
    '',
    '# …or create a new one',
    `${os.bin} up --model Qwen/Qwen2.5-7B-Instruct`,
  ].join('\n')
}

function detectOs() {
  if (typeof navigator === 'undefined') return 'macOS'
  const ua = (navigator.userAgent || '').toLowerCase()
  if (ua.includes('windows')) return 'Windows'
  if (ua.includes('linux') && !ua.includes('android')) return 'Linux'
  return 'macOS'
}

const PROMPT = { color: '#8b86ff' }
const COMMENT = { color: '#6b7689' }

export default function RunNode() {
  const [osId, setOsId] = useState('macOS')
  const [copied, setCopied] = useState(false)
  const timer = useRef(null)

  // Pre-select the visitor's OS on mount (kept out of initial state so SSR
  // and the first paint stay deterministic).
  useEffect(() => setOsId(detectOs()), [])
  useEffect(() => () => clearTimeout(timer.current), [])

  const os = OSES.find((o) => o.id === osId) || OSES[0]

  const copy = () => {
    try {
      navigator.clipboard && navigator.clipboard.writeText(copyText(os))
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
            Pick your platform — the installer and the join/create commands update for it. First run sets up a private
            runtime and the right PyTorch automatically; nothing else to install.
          </p>

          {/* OS selector tiles — drive the terminal card on the right */}
          <div
            role="tablist"
            aria-label="Operating system"
            style={{ marginTop: '24px', display: 'flex', flexWrap: 'wrap', gap: '10px' }}
          >
            {OSES.map((o) => {
              const active = o.id === osId
              return (
                <button
                  key={o.id}
                  type="button"
                  role="tab"
                  aria-selected={active}
                  onClick={() => setOsId(o.id)}
                  style={{
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '1px',
                    padding: '9px 16px',
                    border: `1px solid ${active ? 'var(--accent)' : 'var(--border)'}`,
                    borderRadius: '10px',
                    textAlign: 'left',
                    cursor: 'pointer',
                    background: active
                      ? 'color-mix(in srgb, var(--accent) 12%, var(--card-bg))'
                      : 'var(--card-bg)',
                    color: 'var(--text)',
                    minWidth: '108px',
                    transition: 'border-color .15s, background .15s',
                  }}
                >
                  <span style={{ fontSize: '14.5px', fontWeight: 700 }}>{o.label}</span>
                  <span style={{ fontSize: '12px', color: 'var(--muted2)' }}>{o.sub}</span>
                </button>
              )
            })}
          </div>
          <a
            href={RELEASES}
            style={{
              display: 'inline-block',
              marginTop: '12px',
              fontSize: '13px',
              color: 'var(--muted2)',
              textDecoration: 'none',
            }}
          >
            Prefer the raw binaries? Browse the latest release →
          </a>

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

        {/* code card (slate, fixed dark) — content follows the selected OS */}
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
            <span style={{ marginLeft: '6px', fontFamily: mono, fontSize: '11.5px', color: '#6b7689' }}>
              eujeno — node · {os.label}
            </span>
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
          <div
            style={{
              padding: '20px 18px',
              fontFamily: mono,
              fontSize: '12.5px',
              lineHeight: 1.85,
              color: '#dce3ee',
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-all',
            }}
          >
            <div style={COMMENT}>{os.installComment}</div>
            {os.install.map((line, i) => (
              <div key={i}>
                {i === 0 ? <span style={PROMPT}>{os.prompt} </span> : '  '}
                {line}
              </div>
            ))}
            <div style={{ height: '10px' }} />
            <div style={COMMENT}># join an existing network…</div>
            <div>
              <span style={PROMPT}>{os.prompt} </span>
              {os.bin} serve --peers http://SEED:8001 {os.cont}
            </div>
            <div>{'    '}--model Qwen/Qwen2.5-7B-Instruct</div>
            <div style={{ height: '10px' }} />
            <div style={COMMENT}># …or create a new one</div>
            <div>
              <span style={PROMPT}>{os.prompt} </span>
              {os.bin} up --model Qwen/Qwen2.5-7B-Instruct
            </div>
            <div style={{ height: '10px' }} />
            <div style={COMMENT}>
              {'  '}provisioning runtime ... <span style={{ color: '#6ee7a8' }}>ok</span>
            </div>
            <div style={COMMENT}>
              {'  '}torch backend ......... <span style={{ color: '#dce3ee' }}>{os.torch}</span>
            </div>
            <div style={COMMENT}>
              {'  '}claiming layers ....... <span style={{ color: '#dce3ee' }}>decoder:12-24</span>
            </div>
            <div style={{ height: '8px' }} />
            <div>
              <span style={{ color: '#6ee7a8' }}>●</span> serving{' '}
              <span style={COMMENT}>· 0 errors · joined swarm</span>{' '}
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
