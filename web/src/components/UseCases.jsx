import { container, sectionLabel, h2, card, mono } from '../styles.js'

const CASES = [
  {
    tag: '01 · inference',
    title: 'Apps without infra',
    body: 'Ship an AI product without renting a single GPU. Query the swarm directly from your backend.',
  },
  {
    tag: '02 · research',
    title: 'Open-model science',
    body: 'Probe, fine-tune adapters, and study large open models the community keeps online together.',
  },
  {
    tag: '03 · community',
    title: 'Edge & local-first',
    body: 'Pool GPUs across a lab, a co-op, or a region to keep a shared model running close to home.',
  },
  {
    tag: '04 · resilience',
    title: 'Censorship-resistant AI',
    body: "A model with no central host can't be quietly shut off, throttled, or paywalled out from under you.",
  },
]

export default function UseCases() {
  return (
    <section id="cases" style={{ background: 'var(--section-bg)', borderTop: '1px solid var(--border)' }}>
      <div style={{ ...container, padding: 'var(--section-pad) 28px' }}>
        <div style={sectionLabel}>Use cases</div>
        <h2 style={{ ...h2, maxWidth: '600px' }}>What you build on a model with no landlord.</h2>

        <div
          className="grid-4"
          style={{ marginTop: '44px', display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 'var(--gap)' }}
        >
          {CASES.map((c) => (
            <div
              key={c.tag}
              style={{ ...card, padding: '24px', minHeight: '190px', display: 'flex', flexDirection: 'column' }}
            >
              <div
                style={{
                  fontFamily: mono,
                  fontSize: '11px',
                  color: 'var(--accent)',
                  textTransform: 'uppercase',
                  letterSpacing: '0.04em',
                  fontWeight: 500,
                }}
              >
                {c.tag}
              </div>
              <h3 style={{ margin: 'auto 0 8px', fontSize: '18px', fontWeight: 700, color: 'var(--text)' }}>{c.title}</h3>
              <p style={{ margin: 0, fontSize: '13.5px', lineHeight: 1.55, color: 'var(--muted)' }}>{c.body}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
