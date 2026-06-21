import React, { useState, useEffect, useCallback } from 'react'
import { getSettings, putSettings, getNode } from './api.js'
import { ACCENTS } from './theme.js'

// ── Toggle switch ─────────────────────────────────────────────────────────────
function Toggle({ on, onChange }) {
  return (
    <button
      onClick={() => onChange(!on)}
      style={{
        flex: 'none',
        width: '44px',
        height: '25px',
        borderRadius: '999px',
        border: 'none',
        cursor: 'pointer',
        position: 'relative',
        transition: 'background 0.15s',
        background: on ? 'var(--accent,#4f46e5)' : 'var(--border-strong,#d9dce2)',
      }}
    >
      <span style={{
        position: 'absolute',
        top: '3px',
        left: '3px',
        width: '19px',
        height: '19px',
        borderRadius: '50%',
        background: '#fff',
        boxShadow: '0 1px 2px rgba(0,0,0,0.25)',
        transition: 'transform 0.15s',
        transform: on ? 'translateX(19px)' : 'translateX(0)',
      }} />
    </button>
  )
}

// ── Section card wrapper ───────────────────────────────────────────────────────
function Card({ title, children, style: extraStyle }) {
  return (
    <div style={{
      marginTop: '16px',
      border: '1px solid var(--border,#e9ebef)',
      background: 'var(--card-bg,#fff)',
      borderRadius: '14px',
      padding: '22px',
      boxShadow: 'var(--shadow,none)',
      ...extraStyle,
    }}>
      {title && (
        <div style={{ fontSize: '15px', fontWeight: '700', color: 'var(--text,#0e1116)' }}>
          {title}
        </div>
      )}
      {children}
    </div>
  )
}

// ── Field label ───────────────────────────────────────────────────────────────
// Marks a field that only takes effect when the node next launches.
function LaunchBadge() {
  return (
    <span style={{
      marginLeft: '8px',
      fontSize: '10.5px',
      fontWeight: '600',
      color: 'var(--muted2,#7a828e)',
      background: 'var(--section-bg,#f7f8fa)',
      border: '1px solid var(--border,#e9ebef)',
      borderRadius: '999px',
      padding: '1px 8px',
      whiteSpace: 'nowrap',
      verticalAlign: 'middle',
    }}>↻ restart to apply</span>
  )
}

function Label({ children, launch }) {
  return (
    <label style={{
      display: 'block',
      fontSize: '13px',
      fontWeight: '600',
      color: 'var(--muted,#5b6471)',
      marginBottom: '7px',
    }}>
      {children}{launch ? <LaunchBadge /> : null}
    </label>
  )
}

// ── Select ────────────────────────────────────────────────────────────────────
function Select({ value, onChange, children }) {
  return (
    <select
      value={value}
      onChange={onChange}
      style={{
        width: '100%',
        fontSize: '14.5px',
        color: 'var(--text,#0e1116)',
        background: 'var(--card-bg,#fff)',
        border: '1px solid var(--border-strong,#d9dce2)',
        borderRadius: '9px',
        padding: '11px 13px',
        outline: 'none',
        cursor: 'pointer',
        fontFamily: 'inherit',
      }}
    >
      {children}
    </select>
  )
}

// ── Slider with label + value badge ──────────────────────────────────────────
function SliderField({ label, value, min, max, step, unit, onChange, launch }) {
  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '9px' }}>
        <label style={{ fontSize: '13px', fontWeight: '600', color: 'var(--muted,#5b6471)' }}>
          {label}{launch ? <LaunchBadge /> : null}
        </label>
        <span style={{
          fontFamily: "'JetBrains Mono',monospace",
          fontSize: '13px',
          fontWeight: '600',
          color: 'var(--accent,#4f46e5)',
        }}>
          {value}{unit ? ' ' + unit : ''}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={onChange}
        style={{ width: '100%', accentColor: 'var(--accent,#4f46e5)' }}
      />
    </div>
  )
}

// ── Default settings shape ─────────────────────────────────────────────────────
const DEFAULT_SETTINGS = {
  peerId:     '',
  name:       '',
  model:      '',
  layerMode:  'auto',
  maxLayers:  8,
  maxRam:     16,
  port:       9000,
  region:     'eu-west',
  bandwidth:  200,
  autojoin:   true,
  contribute: true,
  inbound:    true,
  telemetry:  false,
}

// ── Main component ────────────────────────────────────────────────────────────
export default function SettingsPage({ T, accent, dark, theme, setTheme, setAccent }) {
  const [s, setS] = useState(DEFAULT_SETTINGS)
  const [copied,  setCopied]  = useState(false)
  const [saved,   setSaved]   = useState(false)
  const [saving,  setSaving]  = useState(false)
  const [loading, setLoading] = useState(true)
  // The model the node actually serves is fixed at launch (serve --model); read it
  // from /api/node rather than letting it be "chosen" here.
  const [servedModel, setServedModel] = useState('')

  // Load settings on mount
  useEffect(() => {
    let cancelled = false
    getSettings()
      .then(data => {
        if (!cancelled) {
          setS(prev => ({ ...prev, ...data }))
          setLoading(false)
        }
      })
      .catch(() => {
        if (!cancelled) setLoading(false)
      })
    getNode()
      .then(n => { if (!cancelled) setServedModel(n.model || '') })
      .catch(() => {})
    return () => { cancelled = true }
  }, [])

  // Patch a single field
  const patch = useCallback((field) => (e) => {
    setS(prev => ({ ...prev, [field]: e.target.value }))
  }, [])

  const patchNum = useCallback((field) => (e) => {
    setS(prev => ({ ...prev, [field]: Number(e.target.value) }))
  }, [])

  const toggle = useCallback((field) => (val) => {
    setS(prev => ({ ...prev, [field]: val }))
  }, [])

  // Copy peer ID
  async function copyPeerId() {
    try {
      await navigator.clipboard.writeText(s.peerId)
    } catch (_) {
      // fallback: execCommand
      const ta = document.createElement('textarea')
      ta.value = s.peerId
      document.body.appendChild(ta)
      ta.select()
      document.execCommand('copy')
      document.body.removeChild(ta)
    }
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  // Save settings
  async function handleSave() {
    if (saving) return
    setSaving(true)
    try {
      await putSettings(s)
    } catch (_) {
      // best-effort
    } finally {
      setSaving(false)
      setSaved(true)
      setTimeout(() => setSaved(false), 2000)
    }
  }

  const TOGGLES = [
    { key: 'autojoin',   label: 'Auto-join network',      desc: 'Automatically connect to the swarm when the node starts.', launch: true },
    { key: 'contribute', label: 'Contribute layers',       desc: 'Donate GPU/CPU layers to the shared model serving pool.', launch: true },
    { key: 'inbound',    label: 'Accept inbound traffic',  desc: 'Allow other nodes to connect directly to yours.', launch: true },
    { key: 'telemetry',  label: 'Share anonymous metrics', desc: 'Send anonymous performance stats to help improve the network.' },
  ]

  const isDark = theme === 'dark'

  return (
    <div style={{ flex: 1, minHeight: 0, overflowY: 'auto' }}>
      <div style={{ maxWidth: '760px', margin: '0 auto', padding: '30px 36px 60px' }}>

        {/* Page header */}
        <h1 style={{ margin: 0, fontSize: '30px', fontWeight: '800', letterSpacing: '-0.02em', color: 'var(--text,#0e1116)' }}>
          Settings
        </h1>
        <p style={{ margin: '7px 0 0', fontSize: '15px', color: 'var(--muted,#5b6471)' }}>
          Configure how your node joins and serves the network.
        </p>

        {/* ── Identity ── */}
        <Card title="Identity" style={{ marginTop: '26px' }}>
          <div style={{ marginTop: '18px', display: 'flex', flexDirection: 'column', gap: '16px' }}>

            {/* Peer ID */}
            <div>
              <Label>Peer ID</Label>
              <div style={{ display: 'flex', gap: '8px' }}>
                <div style={{
                  flex: '1',
                  fontFamily: "'JetBrains Mono',monospace",
                  fontSize: '13.5px',
                  color: 'var(--text,#0e1116)',
                  background: 'var(--section-bg,#f7f8fa)',
                  border: '1px solid var(--border,#e9ebef)',
                  borderRadius: '9px',
                  padding: '11px 13px',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                }}>
                  {loading ? '—' : (s.peerId || '—')}
                </div>
                <button
                  onClick={copyPeerId}
                  style={{
                    flex: 'none',
                    border: '1px solid var(--border-strong,#d9dce2)',
                    background: 'var(--card-bg,#fff)',
                    color: copied ? 'var(--accent,#4f46e5)' : 'var(--text,#0e1116)',
                    fontSize: '13px',
                    fontWeight: '600',
                    padding: '0 16px',
                    borderRadius: '9px',
                    cursor: 'pointer',
                    fontFamily: 'inherit',
                    transition: 'color 0.15s, border-color 0.15s',
                    whiteSpace: 'nowrap',
                  }}
                >
                  {copied ? 'Copied' : 'Copy'}
                </button>
              </div>
            </div>

            {/* Node name */}
            <div>
              <Label>Node name</Label>
              <input
                type="text"
                value={s.name}
                onChange={patch('name')}
                placeholder="e.g. my-laptop-node"
                style={{
                  width: '100%',
                  boxSizing: 'border-box',
                  fontSize: '14.5px',
                  color: 'var(--text,#0e1116)',
                  background: 'var(--card-bg,#fff)',
                  border: '1px solid var(--border-strong,#d9dce2)',
                  borderRadius: '9px',
                  padding: '11px 13px',
                  outline: 'none',
                  fontFamily: 'inherit',
                }}
                onFocus={e => { e.target.style.borderColor = 'var(--accent,#4f46e5)' }}
                onBlur={e => { e.target.style.borderColor = 'var(--border-strong,#d9dce2)' }}
              />
            </div>
          </div>
        </Card>

        {/* ── Node ── */}
        <Card title="Node">
          <div style={{ marginTop: '18px', display: 'flex', flexDirection: 'column', gap: '18px' }}>

            {/* Model — the served model is fixed at node launch (serve --model); read-only */}
            <div>
              <Label>Model</Label>
              <div style={{
                fontFamily: "'JetBrains Mono',monospace",
                fontSize: '13.5px',
                color: 'var(--text,#0e1116)',
                background: 'var(--section-bg,#f7f8fa)',
                border: '1px solid var(--border,#e9ebef)',
                borderRadius: '9px',
                padding: '11px 13px',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
              }}>
                {loading ? '—' : (servedModel || '—')}
              </div>
              <div style={{ marginTop: '6px', fontSize: '12px', color: 'var(--muted2,#7a828e)' }}>
                Set when the node is launched (<code>serve --model</code>); restart the node to change it.
              </div>
            </div>

            {/* Layer assignment segmented control */}
            <div>
              <Label launch>Layer assignment</Label>
              <div style={{
                display: 'inline-flex',
                padding: '3px',
                background: 'var(--section-bg,#f7f8fa)',
                border: '1px solid var(--border,#e9ebef)',
                borderRadius: '10px',
                gap: '3px',
              }}>
                {['auto', 'manual'].map(mode => {
                  const active = s.layerMode === mode
                  return (
                    <button
                      key={mode}
                      onClick={() => setS(prev => ({ ...prev, layerMode: mode }))}
                      style={{
                        border: 'none',
                        cursor: 'pointer',
                        fontSize: '13.5px',
                        fontWeight: '600',
                        padding: '8px 18px',
                        borderRadius: '7px',
                        fontFamily: 'inherit',
                        transition: 'background 0.12s, color 0.12s',
                        background: active ? 'var(--card-bg,#fff)' : 'transparent',
                        color: active ? 'var(--accent,#4f46e5)' : 'var(--muted,#5b6471)',
                        boxShadow: active ? '0 1px 3px rgba(0,0,0,0.1)' : 'none',
                      }}
                    >
                      {mode === 'auto' ? 'Auto' : 'Manual'}
                    </button>
                  )
                })}
              </div>
            </div>

            {/* Max layers slider */}
            <SliderField
              label="Max layers to host"
              value={s.maxLayers}
              min={2}
              max={20}
              step={1}
              onChange={patchNum('maxLayers')}
              launch
            />

            {/* Max RAM slider */}
            <SliderField
              label={'Max RAM — ' + s.maxRam + ' GB'}
              value={s.maxRam}
              min={4}
              max={48}
              step={2}
              unit="GB"
              onChange={patchNum('maxRam')}
              launch
            />

            {/* Public port */}
            <div>
              <Label launch>Public port</Label>
              <input
                type="number"
                value={s.port}
                onChange={patchNum('port')}
                style={{
                  width: '160px',
                  fontFamily: "'JetBrains Mono',monospace",
                  fontSize: '14px',
                  color: 'var(--text,#0e1116)',
                  background: 'var(--card-bg,#fff)',
                  border: '1px solid var(--border-strong,#d9dce2)',
                  borderRadius: '9px',
                  padding: '11px 13px',
                  outline: 'none',
                }}
                onFocus={e => { e.target.style.borderColor = 'var(--accent,#4f46e5)' }}
                onBlur={e => { e.target.style.borderColor = 'var(--border-strong,#d9dce2)' }}
              />
            </div>
          </div>
        </Card>

        {/* ── Network ── */}
        <Card title="Network">
          <div style={{ marginTop: '18px', display: 'flex', flexDirection: 'column', gap: '18px' }}>

            {/* Region */}
            <div>
              <Label>Region</Label>
              <Select value={s.region} onChange={patch('region')}>
                <option value="eu-west">eu-west</option>
                <option value="eu-north">eu-north</option>
                <option value="us-east">us-east</option>
                <option value="us-west">us-west</option>
                <option value="ap-south">ap-south</option>
                <option value="sa-east">sa-east</option>
              </Select>
            </div>

            {/* Bandwidth slider */}
            <SliderField
              label="Bandwidth limit"
              value={s.bandwidth}
              min={50}
              max={1000}
              step={50}
              unit="Mbps"
              onChange={patchNum('bandwidth')}
              launch
            />
          </div>
        </Card>

        {/* ── Privacy & behaviour toggles ── */}
        <div style={{
          marginTop: '16px',
          border: '1px solid var(--border,#e9ebef)',
          background: 'var(--card-bg,#fff)',
          borderRadius: '14px',
          padding: '8px 22px',
          boxShadow: 'var(--shadow,none)',
        }}>
          {TOGGLES.map((t, i) => {
            const isLast = i === TOGGLES.length - 1
            return (
              <div
                key={t.key}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  gap: '20px',
                  padding: '16px 0',
                  borderBottom: isLast ? 'none' : '1px solid var(--border,#e9ebef)',
                }}
              >
                <div>
                  <div style={{ fontSize: '14.5px', fontWeight: '600', color: 'var(--text,#0e1116)' }}>
                    {t.label}{t.launch ? <LaunchBadge /> : null}
                  </div>
                  <div style={{ fontSize: '13px', color: 'var(--muted2,#7a828e)', marginTop: '2px' }}>
                    {t.desc}
                  </div>
                </div>
                <Toggle on={s[t.key]} onChange={toggle(t.key)} />
              </div>
            )
          })}
        </div>

        {/* ── Save ── */}
        <div style={{ marginTop: '22px', display: 'flex', alignItems: 'center', gap: '14px' }}>
          <button
            onClick={handleSave}
            disabled={saving}
            style={{
              background: 'var(--accent,#4f46e5)',
              color: '#fff',
              fontWeight: '600',
              fontSize: '14.5px',
              padding: '12px 24px',
              border: 'none',
              borderRadius: '10px',
              cursor: saving ? 'not-allowed' : 'pointer',
              fontFamily: 'inherit',
              opacity: saving ? 0.7 : 1,
              boxShadow: '0 8px 18px -8px color-mix(in srgb, var(--accent,#4f46e5) 55%, transparent)',
            }}
          >
            {saving ? 'Saving…' : 'Save changes'}
          </button>

          {saved && (
            <span style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '7px',
              fontSize: '13.5px',
              fontWeight: '600',
              color: '#16a34a',
            }}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
                <path d="M5 12.5 10 17 19 7" />
              </svg>
              Saved
            </span>
          )}
        </div>

        {/* ── Theme & appearance ── */}
        <Card title="Theme &amp; appearance" style={{ marginTop: '32px' }}>
          <div style={{ marginTop: '18px', display: 'flex', flexDirection: 'column', gap: '20px' }}>

            {/* Light / dark toggle */}
            <div style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: '20px',
            }}>
              <div>
                <div style={{ fontSize: '14.5px', fontWeight: '600', color: 'var(--text,#0e1116)' }}>
                  Dark mode
                </div>
                <div style={{ fontSize: '13px', color: 'var(--muted2,#7a828e)', marginTop: '2px' }}>
                  Switch between light and dark interface.
                </div>
              </div>
              <Toggle
                on={isDark}
                onChange={(v) => setTheme(v ? 'dark' : 'light')}
              />
            </div>

            {/* Accent swatches */}
            <div>
              <div style={{ fontSize: '14.5px', fontWeight: '600', color: 'var(--text,#0e1116)', marginBottom: '12px' }}>
                Accent colour
              </div>
              <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
                {ACCENTS.map(color => {
                  const selected = accent === color
                  return (
                    <button
                      key={color}
                      onClick={() => setAccent(color)}
                      title={color}
                      style={{
                        width: '34px',
                        height: '34px',
                        borderRadius: '50%',
                        border: selected ? '3px solid ' + color : '3px solid transparent',
                        outline: selected ? '2px solid var(--card-bg,#fff)' : 'none',
                        outlineOffset: '-5px',
                        background: color,
                        cursor: 'pointer',
                        padding: 0,
                        boxShadow: selected ? '0 0 0 2px ' + color : '0 1px 3px rgba(0,0,0,0.15)',
                        transition: 'box-shadow 0.15s, transform 0.1s',
                        transform: selected ? 'scale(1.15)' : 'scale(1)',
                      }}
                    />
                  )
                })}
              </div>
            </div>
          </div>
        </Card>

      </div>
    </div>
  )
}
