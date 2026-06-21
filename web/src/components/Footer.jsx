import { container, mono } from '../styles.js'
import LogoMark from './LogoMark.jsx'

export default function Footer() {
  return (
    <footer style={{ borderTop: '1px solid var(--border)' }}>
      <div
        style={{
          ...container,
          padding: '26px 28px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: '24px',
          flexWrap: 'wrap',
          fontSize: '13.5px',
          color: 'var(--muted2)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '9px' }}>
          <LogoMark size={22} dot={3.5} gap={2} radius="6px" />
          <span style={{ color: 'var(--text)', fontWeight: 700 }}>Eujeno</span>
        </div>
        <div style={{ display: 'flex', gap: '24px', fontWeight: 500 }}>
          <a className="footer-link" href="docs/" style={{ color: 'var(--muted2)', textDecoration: 'none' }}>
            Docs
          </a>
          <a className="footer-link" href="#run" style={{ color: 'var(--muted2)', textDecoration: 'none' }}>
            Run a node
          </a>
          <a className="footer-link" href="#why" style={{ color: 'var(--muted2)', textDecoration: 'none' }}>
            Protocol
          </a>
        </div>
        <span style={{ fontFamily: mono, fontSize: '12px' }}>peer-to-peer model hosting</span>
      </div>
    </footer>
  )
}
