// Small shared style fragments reused across sections, to keep the ported
// inline styles DRY. Everything else stays local to each component.

export const mono = "'JetBrains Mono',monospace"

export const container = { maxWidth: '1120px', margin: '0 auto' }

export const sectionLabel = {
  fontFamily: mono,
  fontSize: '12.5px',
  letterSpacing: '0.06em',
  color: 'var(--accent)',
  textTransform: 'uppercase',
  fontWeight: 500,
}

export const h2 = {
  margin: '12px 0 0',
  fontSize: 'calc(40px * var(--scale,1))',
  fontWeight: 800,
  letterSpacing: '-0.025em',
  color: 'var(--text)',
  lineHeight: 1.06,
}

export const card = {
  border: '1px solid var(--border)',
  background: 'var(--card-bg)',
  borderRadius: 'var(--radius)',
  boxShadow: 'var(--card-shadow)',
}
