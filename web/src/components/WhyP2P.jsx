import { container, sectionLabel, h2, card, mono } from '../styles.js'

const FEATURES = [
  {
    icon: '⌗',
    title: "Run models you can't host",
    body: 'A 70B model needs no 70B machine. Contribute the layers your GPU can fit and reach the whole thing through the swarm.',
  },
  {
    icon: '⟳',
    title: 'Self-healing routes',
    body: 'Nodes join and drop constantly. When a peer disappears, requests reroute through another holding the same layers. No downtime.',
  },
  {
    icon: '⚇',
    title: 'Permissionless',
    body: 'Anyone can join, host layers, or query the model. No accounts, no gatekeepers, no central API to revoke your access.',
  },
  {
    icon: '≋',
    title: 'Idle compute, reused',
    body: 'Built from spare consumer and prosumer GPUs — inference at a fraction of cloud cost, paid in contributed compute.',
  },
  {
    icon: '◇',
    title: 'One shared model',
    body: 'Every node points at the same weights. Upgrade once and the whole network serves the new model — no redeploy, no fork.',
  },
  {
    icon: '⊞',
    title: 'Scales sideways',
    body: 'More peers means more replicas per layer, more parallel requests, more throughput. Capacity grows with the crowd.',
  },
]

export default function WhyP2P() {
  return (
    <section
      id="why"
      style={{
        background: 'var(--section-bg)',
        borderTop: '1px solid var(--border)',
        borderBottom: '1px solid var(--border)',
      }}
    >
      <div style={{ ...container, padding: 'var(--section-pad) 28px' }}>
        <div style={sectionLabel}>Why peer-to-peer</div>
        <h2 style={{ ...h2, maxWidth: '560px' }}>No data center. No single owner. No off switch.</h2>

        <div
          className="grid-3"
          style={{ marginTop: '44px', display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 'var(--gap)' }}
        >
          {FEATURES.map((f) => (
            <div key={f.title} style={{ ...card, padding: '24px' }}>
              <div
                style={{
                  width: '36px',
                  height: '36px',
                  borderRadius: 'var(--radius-tile)',
                  background: 'color-mix(in srgb, var(--accent) 13%, transparent)',
                  color: 'var(--accent)',
                  display: 'grid',
                  placeItems: 'center',
                  fontSize: '17px',
                  fontFamily: mono,
                }}
              >
                {f.icon}
              </div>
              <h3 style={{ margin: '16px 0 7px', fontSize: '17.5px', fontWeight: 700, color: 'var(--text)' }}>
                {f.title}
              </h3>
              <p style={{ margin: 0, fontSize: '14px', lineHeight: 1.6, color: 'var(--muted)' }}>{f.body}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
