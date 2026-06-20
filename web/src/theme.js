// Default render of the DivMagic template (theme=light, accent=#4f46e5,
// cornerStyle=rounded, density=comfortable, cardShadows=soft, ctaStyle=accent,
// sectionTint=true) expressed as CSS custom properties on the root wrapper.
// Every section reads these via var(--*) — change them here to re-theme.

export const accent = '#4f46e5'

export const palette = {
  pageBg: '#ffffff',
  cardBg: '#ffffff',
  text: '#0e1116',
  muted: '#5b6471',
  muted2: '#7a828e',
  border: '#e9ebef',
  borderStrong: '#d9dce2',
  navBg: 'rgba(255,255,255,0.78)',
  tint: '#f7f8fa',
  canvasBg: '#fafbfc',
}

export const rootVars = {
  '--accent': accent,
  '--page-bg': palette.pageBg,
  '--card-bg': palette.cardBg,
  '--text': palette.text,
  '--muted': palette.muted,
  '--muted2': palette.muted2,
  '--border': palette.border,
  '--border-strong': palette.borderStrong,
  '--nav-bg': palette.navBg,
  '--section-bg': palette.tint,
  '--canvas-bg': palette.canvasBg,
  '--radius': '18px',
  '--radius-sm': '11px',
  '--radius-tile': '10px',
  '--radius-pill': '999px',
  '--section-pad': '72px',
  '--gap': '18px',
  '--scale': 1,
  '--card-shadow': '0 1px 2px rgba(16,24,40,0.05)',
  '--float-shadow': '0 24px 60px -32px rgba(16,24,40,0.22)',
}
