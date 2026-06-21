import React, { useState, useEffect } from 'react'
import { getPeers, restart } from './api.js'
import SwarmCanvas from './SwarmCanvas.jsx'

// Format uptime in seconds as "Xd Yh"
function fmtUptime(sec) {
  if (!sec && sec !== 0) return '—'
  const d = Math.floor(sec / 86400)
  const h = Math.floor((sec % 86400) / 3600)
  return `${d}d ${h}h`
}

// Format a number with fallback
function fmt(v, fallback = '—') {
  if (v == null) return fallback
  return v
}

// Status pill color for online/syncing/offline
function statusColor(status) {
  if (status === 'online' || status === 'serving') return '#16a34a'
  if (status === 'syncing') return '#d97706'
  return '#9aa3b2'
}

function statusLabel(status) {
  if (status === 'online' || status === 'serving') return 'online'
  if (status === 'syncing') return 'syncing'
  return 'offline'
}

// Stat card
function StatCard({ label, value, unit, valueColor }) {
  return (
    <div style={{
      border: '1px solid var(--border,#e9ebef)',
      background: 'var(--card-bg,#fff)',
      borderRadius: '13px',
      padding: '18px',
      boxShadow: 'var(--shadow,none)',
    }}>
      <div style={{
        fontFamily: "'JetBrains Mono',monospace",
        fontSize: '11px',
        color: 'var(--muted2,#7a828e)',
        textTransform: 'uppercase',
        letterSpacing: '0.04em',
      }}>{label}</div>
      <div style={{
        marginTop: '8px',
        fontSize: '28px',
        fontWeight: '800',
        letterSpacing: '-0.02em',
        color: valueColor || 'var(--text,#0e1116)',
      }}>
        {value}
        {unit && <span style={{ fontSize: '14px', fontWeight: '600', color: 'var(--muted2,#7a828e)', marginLeft: '4px' }}>{unit}</span>}
      </div>
    </div>
  )
}

export default function NetworkPage({ T, accent, dark, node, metrics }) {
  const [peers, setPeers] = useState(null)
  const [restartMsg, setRestartMsg] = useState(null)

  // Poll peers every 3s
  useEffect(() => {
    let cancelled = false
    async function poll() {
      try {
        const data = await getPeers()
        if (!cancelled) setPeers(data?.peers ?? [])
      } catch (_) { /* best-effort */ }
    }
    poll()
    const id = setInterval(poll, 3000)
    return () => { cancelled = true; clearInterval(id) }
  }, [])

  async function handleRestart() {
    try {
      await restart()
      setRestartMsg('Restarting…')
      setTimeout(() => setRestartMsg(null), 3000)
    } catch (_) {
      setRestartMsg('Restart failed')
      setTimeout(() => setRestartMsg(null), 3000)
    }
  }

  const isOnline = node?.status === 'serving' || node?.status === 'online'
  // Real swarm size = this node + its connected peers (not a decorative minimum).
  const swarmCount = (metrics?.connectedPeers ?? 0) + 1

  return (
    <div style={{ flex: 1, minHeight: 0, overflowY: 'auto' }}>
      <div style={{ maxWidth: '1080px', margin: '0 auto', padding: '30px 36px 56px' }}>

        {/* Header */}
        <div style={{
          display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between',
          gap: '20px', flexWrap: 'wrap',
        }}>
          <div>
            <h1 style={{ margin: 0, fontSize: '30px', fontWeight: '800', letterSpacing: '-0.02em', color: 'var(--text,#0e1116)' }}>
              Network
            </h1>
            <p style={{ margin: '7px 0 0', fontSize: '15px', color: 'var(--muted,#5b6471)' }}>
              You're serving{' '}
              <span style={{ fontFamily: "'JetBrains Mono',monospace", color: 'var(--text,#0e1116)' }}>
                {node?.model ?? '—'}
              </span>{' '}
              with the swarm.
            </p>
          </div>
          <span style={{
            display: 'inline-flex', alignItems: 'center', gap: '8px',
            fontSize: '13px', fontWeight: '600',
            color: isOnline ? '#16a34a' : '#9aa3b2',
            background: isOnline ? 'rgba(22,163,74,0.1)' : 'rgba(154,163,178,0.1)',
            border: `1px solid ${isOnline ? 'rgba(22,163,74,0.25)' : 'rgba(154,163,178,0.25)'}`,
            padding: '7px 13px', borderRadius: '999px',
          }}>
            <span style={{
              width: '7px', height: '7px', borderRadius: '50%',
              background: isOnline ? '#16a34a' : '#9aa3b2',
              animation: isOnline ? 'blink 1.8s steps(2) infinite' : 'none',
            }} />
            {isOnline ? 'Online' : (node ? 'Offline' : 'Loading…')}
          </span>
        </div>

        {/* Stat cards — 5 cards */}
        <div style={{
          marginTop: '24px',
          display: 'grid',
          gridTemplateColumns: 'repeat(5,1fr)',
          gap: '14px',
        }}>
          <StatCard label="Connected peers" value={fmt(metrics?.connectedPeers, '—')} />
          <StatCard label="Your layers" value={fmt(node?.layers, '—')} valueColor={accent} />
          <StatCard label="Throughput" value={fmt(metrics?.throughputTokS, '—')} unit="tok/s" />
          <StatCard label="Avg latency" value={fmt(metrics?.avgLatencyMs, '—')} unit="ms" />
          <StatCard label="Requests served" value={fmt(metrics?.requestsServed, '—')} />
        </div>

        {/* Canvas + Your node */}
        <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: '1fr 300px', gap: '14px' }}>

          {/* Swarm canvas card */}
          <div style={{
            border: '1px solid var(--border,#e9ebef)',
            background: 'var(--card-bg,#fff)',
            borderRadius: '14px',
            overflow: 'hidden',
            boxShadow: 'var(--shadow,none)',
          }}>
            <div style={{
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              padding: '13px 16px', borderBottom: '1px solid var(--border,#eef0f3)',
            }}>
              <span style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: '12px', color: 'var(--muted2,#7a828e)' }}>
                swarm.topology
              </span>
              <span style={{
                display: 'inline-flex', alignItems: 'center', gap: '7px',
                fontFamily: "'JetBrains Mono',monospace", fontSize: '11px', color: accent,
              }}>
                <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: accent }} />
                {fmt(metrics?.activeQueries, 0)} active queries
              </span>
            </div>
            <SwarmCanvas accent={accent} dark={dark} count={swarmCount} />
          </div>

          {/* Your node panel */}
          <div style={{
            border: '1px solid var(--border,#e9ebef)',
            background: 'var(--card-bg,#fff)',
            borderRadius: '14px',
            padding: '18px',
            boxShadow: 'var(--shadow,none)',
            display: 'flex',
            flexDirection: 'column',
          }}>
            <div style={{
              fontSize: '13px', fontWeight: '700', textTransform: 'uppercase',
              letterSpacing: '0.05em', color: 'var(--muted2,#7a828e)',
            }}>
              Your node
            </div>

            <div style={{ marginTop: '14px', display: 'flex', flexDirection: 'column', gap: 0 }}>
              {[
                { label: 'Status', value: node?.status ?? '—', valueStyle: { fontWeight: '600', color: statusColor(node?.status) } },
                { label: 'Layers', value: node?.layers ?? '—', valueStyle: { fontFamily: "'JetBrains Mono',monospace", fontWeight: '600' } },
                { label: 'RAM used', value: node ? `${fmt(node.ramUsedGb, '—')} / ${fmt(node.ramTotalGb, '—')} GB` : '—', valueStyle: { fontWeight: '600' } },
                { label: 'Region', value: node?.region ?? '—', valueStyle: { fontWeight: '600' } },
                { label: 'Uptime', value: fmtUptime(node?.uptimeSec), valueStyle: { fontWeight: '600' } },
                { label: 'Requests served', value: fmt(node?.requestsServed, '—'), valueStyle: { fontWeight: '600' }, last: true },
              ].map(({ label, value, valueStyle, last }, i) => (
                <div key={i} style={{
                  display: 'flex', justifyContent: 'space-between',
                  padding: '9px 0',
                  borderBottom: last ? 'none' : '1px solid var(--border,#eef0f3)',
                  fontSize: '13.5px',
                }}>
                  <span style={{ color: 'var(--muted,#5b6471)' }}>{label}</span>
                  <span style={valueStyle}>{value}</span>
                </div>
              ))}
            </div>

            <div style={{ flex: 1 }} />
            <button
              onClick={handleRestart}
              style={{
                marginTop: '16px', width: '100%',
                border: '1px solid var(--border-strong,#d9dce2)',
                background: 'var(--card-bg,#fff)',
                color: 'var(--text,#0e1116)',
                fontWeight: '600', fontSize: '13.5px',
                padding: '10px', borderRadius: '9px', cursor: 'pointer',
              }}
            >
              {restartMsg ?? 'Restart node'}
            </button>
          </div>
        </div>

        {/* Peers table */}
        <div style={{
          marginTop: '14px',
          border: '1px solid var(--border,#e9ebef)',
          background: 'var(--card-bg,#fff)',
          borderRadius: '14px',
          overflow: 'hidden',
          boxShadow: 'var(--shadow,none)',
        }}>
          <div style={{ padding: '14px 18px', borderBottom: '1px solid var(--border,#eef0f3)', fontSize: '14px', fontWeight: '700', color: 'var(--text,#0e1116)' }}>
            Peers in your chain
          </div>

          {/* Table header */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: '1.4fr 1fr 1fr 0.8fr 0.9fr',
            padding: '10px 18px',
            borderBottom: '1px solid var(--border,#eef0f3)',
            fontFamily: "'JetBrains Mono',monospace",
            fontSize: '11px',
            color: 'var(--muted2,#7a828e)',
            textTransform: 'uppercase',
            letterSpacing: '0.04em',
          }}>
            <span>Peer</span><span>Layers</span><span>Region</span><span>Latency</span><span>Status</span>
          </div>

          {/* Table rows */}
          {peers == null ? (
            <div style={{ padding: '18px', fontSize: '13px', color: 'var(--muted2,#7a828e)' }}>Loading peers…</div>
          ) : peers.length === 0 ? (
            <div style={{ padding: '18px', fontSize: '13px', color: 'var(--muted2,#7a828e)' }}>No peers connected yet.</div>
          ) : (
            peers.map((peer, i) => {
              const isLast = i === peers.length - 1
              const sc = statusColor(peer.status)
              return (
                <div key={peer.peerId ?? i} style={{
                  display: 'grid',
                  gridTemplateColumns: '1.4fr 1fr 1fr 0.8fr 0.9fr',
                  padding: '13px 18px',
                  borderBottom: isLast ? 'none' : '1px solid var(--border,#eef0f3)',
                  alignItems: 'center',
                  fontSize: '13.5px',
                }}>
                  <span style={{ fontFamily: "'JetBrains Mono',monospace", color: 'var(--text,#0e1116)' }}>
                    {peer.peerId ? `node·${peer.peerId.slice(0, 6)}` : '—'}
                  </span>
                  <span style={{ fontFamily: "'JetBrains Mono',monospace", color: 'var(--muted,#5b6471)' }}>
                    {peer.layers ?? '—'}
                  </span>
                  <span style={{ color: 'var(--muted,#5b6471)' }}>{peer.region ?? '—'}</span>
                  <span style={{ color: 'var(--muted,#5b6471)' }}>
                    {peer.latencyMs != null ? `${peer.latencyMs} ms` : '—'}
                  </span>
                  <span style={{ display: 'inline-flex', alignItems: 'center', gap: '7px', color: sc, fontWeight: '600' }}>
                    <span style={{ width: '7px', height: '7px', borderRadius: '50%', background: sc }} />
                    {statusLabel(peer.status)}
                  </span>
                </div>
              )
            })
          )}
        </div>

      </div>
    </div>
  )
}
