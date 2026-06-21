import { container } from '../styles.js'
import LogoMark from './LogoMark.jsx'

export default function Nav() {
  return (
    <header
      style={{
        position: 'sticky',
        top: 0,
        zIndex: 70,
        backdropFilter: 'blur(12px)',
        WebkitBackdropFilter: 'blur(12px)',
        background: 'var(--nav-bg)',
        borderBottom: '1px solid var(--border)',
      }}
    >
      <div
        style={{
          ...container,
          padding: '14px 28px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: '24px',
        }}
      >
        <a
          href="#top"
          style={{ display: 'flex', alignItems: 'center', gap: '10px', textDecoration: 'none', color: 'var(--text)' }}
        >
          <LogoMark />
          <span style={{ fontWeight: 700, letterSpacing: '-0.01em', fontSize: '18px' }}>Eujeno</span>
        </a>
        <nav style={{ display: 'flex', alignItems: 'center', gap: '28px', fontSize: '14.5px', fontWeight: 500 }}>
          <a className="nav-link nav-extra" href="#how" style={{ color: 'var(--muted)', textDecoration: 'none' }}>
            How it works
          </a>
          <a className="nav-link nav-extra" href="#why" style={{ color: 'var(--muted)', textDecoration: 'none' }}>
            Why P2P
          </a>
          <a className="nav-link nav-extra" href="#cases" style={{ color: 'var(--muted)', textDecoration: 'none' }}>
            Use cases
          </a>
          <a className="nav-link nav-extra" href="docs/" style={{ color: 'var(--muted)', textDecoration: 'none' }}>
            Docs
          </a>
          <a
            href="#run"
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '7px',
              color: '#fff',
              background: 'var(--accent)',
              textDecoration: 'none',
              fontWeight: 600,
              fontSize: '14px',
              padding: '9px 16px',
              borderRadius: 'var(--radius-sm)',
            }}
          >
            Run a node
          </a>
        </nav>
      </div>
    </header>
  )
}
