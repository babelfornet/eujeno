import { container, sectionLabel, h2, card, mono } from '../styles.js'

const STEPS = [
  {
    n: '1',
    title: 'The model is sharded',
    body: "A model's transformer blocks are split into contiguous layer ranges. Each range is a self-contained stage of the forward pass.",
  },
  {
    n: '2',
    title: 'Nodes claim layers',
    body: 'Each peer hosts whatever layers it can fit in VRAM and announces them. The swarm self-organizes into a full end-to-end chain.',
  },
  {
    n: '3',
    title: 'Anyone queries it',
    body: 'A request routes hop-by-hop through the chain — activations passed peer to peer — and the final node streams tokens back.',
  },
]

const STAGES = [
  { node: 'node·7f3a', range: 'L0–15', kind: 'embed + attn', accent: true },
  { node: 'node·a19c', range: 'L16–31', kind: 'transformer', accent: false },
  { node: 'node·c4e1', range: 'L32–47', kind: 'transformer', accent: false },
  { node: 'node·d8b0', range: 'L48–63', kind: 'norm + head', accent: true },
]

const arrow = (
  <div style={{ flex: 'none', alignSelf: 'center', color: 'var(--muted2)', padding: '0 8px' }}>→</div>
)

export default function HowItWorks() {
  return (
    <section id="how" style={{ ...container, padding: 'var(--section-pad) 28px' }}>
      <div style={sectionLabel}>How it works</div>
      <h2 style={{ ...h2, maxWidth: '600px' }}>A model assembled from machines that never meet.</h2>

      <div
        className="grid-3"
        style={{ marginTop: '44px', display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 'var(--gap)' }}
      >
        {STEPS.map((s) => (
          <div key={s.n} style={{ ...card, padding: '26px' }}>
            <div
              style={{
                width: '38px',
                height: '38px',
                borderRadius: 'var(--radius-tile)',
                background: 'color-mix(in srgb, var(--accent) 13%, transparent)',
                color: 'var(--accent)',
                display: 'grid',
                placeItems: 'center',
                fontFamily: mono,
                fontWeight: 600,
                fontSize: '15px',
              }}
            >
              {s.n}
            </div>
            <h3 style={{ margin: '18px 0 8px', fontSize: '19px', fontWeight: 700, color: 'var(--text)' }}>{s.title}</h3>
            <p style={{ margin: 0, fontSize: '14.5px', lineHeight: 1.6, color: 'var(--muted)' }}>{s.body}</p>
          </div>
        ))}
      </div>

      {/* pipeline */}
      <div
        style={{
          marginTop: '20px',
          border: '1px solid var(--border)',
          background: 'var(--section-bg)',
          borderRadius: 'var(--radius)',
          padding: '30px 26px',
          position: 'relative',
          overflow: 'hidden',
        }}
      >
        <div
          style={{
            position: 'absolute',
            top: 0,
            bottom: 0,
            left: 0,
            width: '90px',
            background: 'linear-gradient(90deg, transparent, color-mix(in srgb, var(--accent) 14%, transparent), transparent)',
            animation: 'flow 5s linear infinite',
            pointerEvents: 'none',
          }}
        />
        <div style={{ position: 'relative', display: 'flex', alignItems: 'stretch', flexWrap: 'nowrap', overflowX: 'auto' }}>
          <div
            style={{
              flex: 'none',
              alignSelf: 'center',
              fontFamily: mono,
              fontSize: '12px',
              fontWeight: 500,
              color: 'var(--muted)',
              padding: '13px 15px',
              border: '1px dashed var(--border-strong)',
              borderRadius: 'var(--radius-sm)',
              background: 'var(--card-bg)',
            }}
          >
            prompt
          </div>
          {STAGES.map((st, i) => (
            <div key={st.node} style={{ display: 'contents' }}>
              {arrow}
              <div
                style={{
                  flex: '1 1 0',
                  minWidth: '140px',
                  border: st.accent
                    ? '1px solid color-mix(in srgb, var(--accent) 40%, transparent)'
                    : '1px solid var(--border-strong)',
                  background: st.accent ? 'color-mix(in srgb, var(--accent) 7%, transparent)' : 'var(--card-bg)',
                  borderRadius: 'var(--radius-sm)',
                  padding: '15px',
                }}
              >
                <div style={{ fontFamily: mono, fontSize: '11px', color: 'var(--muted2)' }}>{st.node}</div>
                <div
                  style={{
                    fontFamily: mono,
                    fontSize: '15px',
                    color: st.accent ? 'var(--accent)' : 'var(--text)',
                    fontWeight: 600,
                    marginTop: '5px',
                  }}
                >
                  {st.range}
                </div>
                <div style={{ fontSize: '12px', color: 'var(--muted2)', marginTop: '6px' }}>{st.kind}</div>
              </div>
              {i === STAGES.length - 1 && arrow}
            </div>
          ))}
          <div
            style={{
              flex: 'none',
              alignSelf: 'center',
              fontFamily: mono,
              fontSize: '12px',
              fontWeight: 600,
              color: '#fff',
              background: 'var(--accent)',
              padding: '13px 15px',
              borderRadius: 'var(--radius-sm)',
            }}
          >
            tokens
          </div>
        </div>
      </div>
    </section>
  )
}
