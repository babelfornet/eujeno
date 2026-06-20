// The 2x2-dot accent square used in the nav and footer.
export default function LogoMark({ size = 24, dot = 4, gap = 2.5, radius = 'calc(var(--radius-tile) * 0.7)' }) {
  const cell = (bg) => ({ width: `${dot}px`, height: `${dot}px`, background: bg, borderRadius: '1px' })
  return (
    <span
      style={{
        width: `${size}px`,
        height: `${size}px`,
        borderRadius: radius,
        background: 'var(--accent)',
        display: 'grid',
        placeItems: 'center',
      }}
    >
      <span style={{ display: 'grid', gridTemplateColumns: `repeat(2,${dot}px)`, gap: `${gap}px` }}>
        <span style={cell('#fff')} />
        <span style={cell('rgba(255,255,255,0.5)')} />
        <span style={cell('rgba(255,255,255,0.5)')} />
        <span style={cell('#fff')} />
      </span>
    </span>
  )
}
