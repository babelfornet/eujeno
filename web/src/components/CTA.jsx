import { container } from '../styles.js'

export default function CTA() {
  return (
    <section style={{ ...container, padding: '80px 28px' }}>
      <div
        style={{
          borderRadius: 'calc(var(--radius) + 6px)',
          background: 'var(--accent)',
          border: '1px solid transparent',
          padding: '64px 40px',
          textAlign: 'center',
          position: 'relative',
          overflow: 'hidden',
          boxShadow: 'var(--float-shadow)',
        }}
      >
        <div
          style={{
            position: 'absolute',
            inset: 0,
            backgroundImage: 'radial-gradient(rgba(255,255,255,0.16) 1px, transparent 1px)',
            backgroundSize: '22px 22px',
            opacity: 0.5,
            pointerEvents: 'none',
          }}
        />
        <h2
          style={{
            position: 'relative',
            margin: '0 auto',
            fontSize: 'calc(46px * var(--scale,1))',
            fontWeight: 800,
            letterSpacing: '-0.025em',
            color: '#fff',
            maxWidth: '600px',
            lineHeight: 1.05,
          }}
        >
          Lend your GPU a few layers.
          <br />
          Get a whole model back.
        </h2>
        <div
          style={{
            position: 'relative',
            marginTop: '30px',
            display: 'flex',
            gap: '12px',
            justifyContent: 'center',
            flexWrap: 'wrap',
          }}
        >
          <a
            href="#run"
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '8px',
              background: '#fff',
              color: 'var(--accent)',
              fontWeight: 700,
              fontSize: '15.5px',
              textDecoration: 'none',
              padding: '14px 28px',
              borderRadius: 'var(--radius-sm)',
            }}
          >
            Run a node
          </a>
          <a
            href="#how"
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '8px',
              background: 'transparent',
              color: '#fff',
              fontWeight: 600,
              fontSize: '15.5px',
              textDecoration: 'none',
              padding: '14px 26px',
              borderRadius: 'var(--radius-sm)',
              border: '1px solid rgba(255,255,255,0.45)',
            }}
          >
            Read the protocol
          </a>
        </div>
      </div>
    </section>
  )
}
