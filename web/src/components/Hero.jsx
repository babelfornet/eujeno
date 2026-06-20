import { container, mono } from '../styles.js'
import { accent } from '../theme.js'
import SwarmCanvas from './SwarmCanvas.jsx'

export default function Hero() {
  return (
    <section
      id="top"
      style={{ position: 'relative', ...container, padding: '84px 28px 56px', textAlign: 'center' }}
    >
      <div style={{ position: 'relative', zIndex: 1 }}>
        <div
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '9px',
            fontSize: '13px',
            fontWeight: 600,
            color: 'var(--accent)',
            border: '1px solid color-mix(in srgb, var(--accent) 26%, transparent)',
            background: 'color-mix(in srgb, var(--accent) 7%, transparent)',
            padding: '6px 13px',
            borderRadius: 'var(--radius-pill)',
          }}
        >
          <span
            style={{
              width: '7px',
              height: '7px',
              background: 'var(--accent)',
              borderRadius: '50%',
              animation: 'blink 1.6s steps(2) infinite',
            }}
          />
          Peer-to-peer AI layer network
        </div>

        <h1
          style={{
            margin: '26px auto 0',
            fontSize: 'calc(66px * var(--scale,1))',
            lineHeight: 1.02,
            fontWeight: 800,
            letterSpacing: '-0.03em',
            color: 'var(--text)',
            maxWidth: '880px',
          }}
        >
          One model, split across
          <br />
          the <span style={{ color: 'var(--accent)' }}>network.</span>
        </h1>

        <p
          style={{
            margin: '24px auto 0',
            fontSize: '19px',
            lineHeight: 1.6,
            color: 'var(--muted)',
            maxWidth: '600px',
            fontWeight: 400,
          }}
        >
          Eujeno shards a large language model layer by layer across independent nodes. No machine holds the whole
          model — together they run it, and anyone on the network can query it.
        </p>

        <div style={{ marginTop: '32px', display: 'flex', gap: '12px', justifyContent: 'center', flexWrap: 'wrap' }}>
          <a
            href="#run"
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '8px',
              background: 'var(--accent)',
              color: '#fff',
              fontWeight: 600,
              fontSize: '15.5px',
              textDecoration: 'none',
              padding: '13px 24px',
              borderRadius: 'var(--radius-sm)',
              boxShadow: '0 8px 20px -8px color-mix(in srgb, var(--accent) 55%, transparent)',
            }}
          >
            Run a node
          </a>
          <a
            className="btn-outline"
            href="#how"
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '8px',
              color: 'var(--text)',
              fontWeight: 600,
              fontSize: '15.5px',
              textDecoration: 'none',
              padding: '13px 24px',
              borderRadius: 'var(--radius-sm)',
              border: '1px solid var(--border-strong)',
              background: 'var(--card-bg)',
            }}
          >
            See how it works
          </a>
        </div>

        {/* network card */}
        <div
          style={{
            marginTop: '56px',
            border: '1px solid var(--border)',
            background: 'var(--card-bg)',
            borderRadius: 'var(--radius)',
            overflow: 'hidden',
            boxShadow: 'var(--float-shadow)',
            textAlign: 'left',
          }}
        >
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '13px 18px',
              borderBottom: '1px solid var(--border)',
            }}
          >
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '10px',
                fontFamily: mono,
                fontSize: '12px',
                color: 'var(--muted2)',
                letterSpacing: '0.02em',
              }}
            >
              <span style={{ display: 'flex', gap: '5px' }}>
                <span style={{ width: '9px', height: '9px', borderRadius: '50%', background: 'var(--border-strong)' }} />
                <span style={{ width: '9px', height: '9px', borderRadius: '50%', background: 'var(--border-strong)' }} />
                <span style={{ width: '9px', height: '9px', borderRadius: '50%', background: 'var(--border-strong)' }} />
              </span>
              swarm.topology
            </div>
            <span
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '7px',
                fontFamily: mono,
                fontSize: '11.5px',
                fontWeight: 500,
                color: 'var(--accent)',
              }}
            >
              <span style={{ width: '7px', height: '7px', background: 'var(--accent)', borderRadius: '50%' }} />
              LIVE
            </span>
          </div>

          <SwarmCanvas
            accent={accent}
            style={{ display: 'block', width: '100%', height: '380px', background: 'var(--canvas-bg)' }}
          />

          <div
            style={{
              display: 'flex',
              gap: '20px',
              padding: '13px 18px',
              borderTop: '1px solid var(--border)',
              fontFamily: mono,
              fontSize: '11.5px',
              color: 'var(--muted2)',
            }}
          >
            <span style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}>
              <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: 'var(--accent)' }} />
              token in transit
            </span>
            <span style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}>
              <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#aab0bb' }} />
              peer node
            </span>
            <span style={{ marginLeft: 'auto' }}>forward pass →</span>
          </div>
        </div>
      </div>
    </section>
  )
}
