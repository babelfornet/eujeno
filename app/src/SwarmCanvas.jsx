import React, { useRef, useEffect } from 'react'

// Port of comp _init/_draw swarm animation (lines 397–442)
export default function SwarmCanvas({ accent, dark, count }) {
  const canvasRef = useRef(null)
  // Keep mutable state in a ref so animation loop always sees latest without re-renders
  const stateRef = useRef({ nodes: null, adj: null, pulses: null, W: 0, H: 0, N: 0, raf: null })

  // Keep accent/dark accessible from the animation loop without re-creating it
  const themeRef = useRef({ accent, dark })
  useEffect(() => { themeRef.current = { accent, dark } }, [accent, dark])

  useEffect(() => {
    function dist(a, b) { return Math.hypot(a.x - b.x, a.y - b.y) }

    function init(N, W, H) {
      const nodes = []
      const pad = 26
      for (let i = 0; i < N; i++) {
        nodes.push({
          x: pad + Math.random() * (W - 2 * pad),
          y: pad + Math.random() * (H - 2 * pad),
          vx: (Math.random() - 0.5) * 0.11,
          vy: (Math.random() - 0.5) * 0.11,
          r: Math.random() * 1.5 + 1.7,
          pulse: 0,
        })
      }
      const adj = nodes.map(() => [])
      for (let i = 0; i < N; i++) {
        const order = nodes
          .map((n, j) => ({ j, d: j === i ? Infinity : dist(nodes[i], n) }))
          .sort((a, b) => a.d - b.d)
        for (let k = 0; k < 2; k++) {
          const j = order[k].j
          if (!adj[i].includes(j)) adj[i].push(j)
          if (!adj[j].includes(i)) adj[j].push(i)
        }
      }
      const pulses = []
      const pcount = Math.max(3, Math.round(N / 5))
      for (let p = 0; p < pcount; p++) {
        const a = Math.floor(Math.random() * N)
        const nb = adj[a]
        if (!nb.length) continue
        pulses.push({
          a,
          b: nb[Math.floor(Math.random() * nb.length)],
          t: Math.random(),
          speed: 0.004 + Math.random() * 0.005,
        })
      }
      const s = stateRef.current
      s.nodes = nodes
      s.adj = adj
      s.pulses = pulses
    }

    function draw() {
      const canvas = canvasRef.current
      if (!canvas) { stateRef.current.raf = requestAnimationFrame(draw); return }
      const ctx = canvas.getContext('2d')
      const rect = canvas.getBoundingClientRect()
      const W = rect.width, H = rect.height
      if (W < 2 || H < 2) { stateRef.current.raf = requestAnimationFrame(draw); return }

      const N = Math.max(1, stateRef.current._count ?? 1)
      const dpr = Math.min(window.devicePixelRatio || 1, 2)
      const s = stateRef.current

      if (W !== s.W || H !== s.H || N !== s.N) {
        s.W = W; s.H = H; s.N = N
        canvas.width = Math.max(1, W * dpr)
        canvas.height = Math.max(1, H * dpr)
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
        init(N, W, H)
      }

      const { accent: ac, dark: dk } = themeRef.current
      const edge = dk ? 'rgba(255,255,255,0.07)' : 'rgba(14,17,22,0.09)'
      const idle = dk ? '#3a4150' : '#aab0bb'
      const { nodes, adj, pulses } = s

      ctx.clearRect(0, 0, W, H)

      // Update positions
      for (const n of nodes) {
        n.x += n.vx; n.y += n.vy
        if (n.x < 18 || n.x > W - 18) n.vx *= -1
        if (n.y < 18 || n.y > H - 18) n.vy *= -1
        if (n.pulse > 0) n.pulse -= 0.03
      }

      // Draw edges
      ctx.lineWidth = 1; ctx.strokeStyle = edge
      for (let i = 0; i < nodes.length; i++) {
        for (const j of adj[i]) {
          if (j <= i) continue
          ctx.beginPath()
          ctx.moveTo(nodes[i].x, nodes[i].y)
          ctx.lineTo(nodes[j].x, nodes[j].y)
          ctx.stroke()
        }
      }

      // Draw pulses
      for (const pl of pulses) {
        pl.t += pl.speed
        if (pl.t >= 1) {
          nodes[pl.b].pulse = 1
          pl.a = pl.b
          const nb = adj[pl.a]
          pl.b = nb.length ? nb[Math.floor(Math.random() * nb.length)] : pl.a
          pl.t = 0
        }
        const A = nodes[pl.a], B = nodes[pl.b]
        const x = A.x + (B.x - A.x) * pl.t
        const y = A.y + (B.y - A.y) * pl.t
        ctx.strokeStyle = ac; ctx.globalAlpha = 0.5; ctx.lineWidth = 1.6
        ctx.beginPath()
        ctx.moveTo(A.x + (B.x - A.x) * Math.max(0, pl.t - 0.18), A.y + (B.y - A.y) * Math.max(0, pl.t - 0.18))
        ctx.lineTo(x, y)
        ctx.stroke()
        ctx.globalAlpha = 1
        ctx.fillStyle = ac; ctx.shadowColor = ac; ctx.shadowBlur = 10
        ctx.beginPath(); ctx.arc(x, y, 2.6, 0, Math.PI * 2); ctx.fill()
        ctx.shadowBlur = 0
      }

      // Draw nodes
      for (const n of nodes) {
        if (n.pulse > 0) {
          ctx.fillStyle = ac; ctx.globalAlpha = n.pulse * 0.4
          ctx.beginPath(); ctx.arc(n.x, n.y, n.r + 7 * n.pulse, 0, Math.PI * 2); ctx.fill()
          ctx.globalAlpha = 1
        }
        ctx.fillStyle = n.pulse > 0.05 ? ac : idle
        ctx.beginPath(); ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2); ctx.fill()
      }

      s.raf = requestAnimationFrame(draw)
    }

    stateRef.current.raf = requestAnimationFrame(draw)

    return () => {
      if (stateRef.current.raf != null) cancelAnimationFrame(stateRef.current.raf)
    }
  }, []) // mount/unmount only; accent/dark via themeRef; count via stateRef._count

  // Propagate count changes -> force re-init on next frame
  useEffect(() => {
    stateRef.current._count = Math.max(1, count ?? 1)
    stateRef.current.N = 0 // triggers re-init in draw loop
  }, [count])

  return (
    <canvas
      ref={canvasRef}
      style={{ display: 'block', width: '100%', height: '340px', background: 'var(--canvas-bg,#fafbfc)' }}
    />
  )
}
