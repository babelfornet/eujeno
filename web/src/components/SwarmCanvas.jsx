import { useEffect, useRef } from 'react'

// Faithful port of the template's DCLogic canvas (_init / _draw): a drifting
// mesh of peer nodes with token pulses travelling along the edges.
export default function SwarmCanvas({
  count = 32,
  speed = 1,
  shape = 'dots',
  glow = true,
  accent = '#4f46e5',
  dark = false,
  style,
}) {
  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')

    let raf
    let W = -1
    let H = -1
    let N = -1
    let nodes = []
    let adj = []
    let pulses = []

    const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y)

    const init = (n, w, h) => {
      const pad = 26
      nodes = []
      for (let i = 0; i < n; i++) {
        nodes.push({
          x: pad + Math.random() * (w - 2 * pad),
          y: pad + Math.random() * (h - 2 * pad),
          vx: (Math.random() - 0.5) * 0.11,
          vy: (Math.random() - 0.5) * 0.11,
          r: Math.random() * 1.5 + 1.7,
          pulse: 0,
        })
      }
      adj = nodes.map(() => [])
      for (let i = 0; i < n; i++) {
        const order = nodes
          .map((node, j) => ({ j, d: j === i ? Infinity : dist(nodes[i], node) }))
          .sort((a, b) => a.d - b.d)
        for (let k = 0; k < 2; k++) {
          const j = order[k].j
          if (!adj[i].includes(j)) adj[i].push(j)
          if (!adj[j].includes(i)) adj[j].push(i)
        }
      }
      pulses = []
      const pc = Math.max(3, Math.round(n / 5))
      for (let p = 0; p < pc; p++) {
        const a = Math.floor(Math.random() * n)
        const nb = adj[a]
        if (!nb.length) continue
        const b = nb[Math.floor(Math.random() * nb.length)]
        pulses.push({ a, b, t: Math.random(), speed: 0.004 + Math.random() * 0.005 })
      }
    }

    const draw = () => {
      const rect = canvas.getBoundingClientRect()
      const w = rect.width
      const h = rect.height
      if (w < 2 || h < 2) {
        raf = requestAnimationFrame(draw)
        return
      }

      const n = Math.max(6, Math.round(count))
      const dpr = Math.min(window.devicePixelRatio || 1, 2)

      if (w !== W || h !== H || n !== N) {
        W = w
        H = h
        N = n
        canvas.width = Math.max(1, w * dpr)
        canvas.height = Math.max(1, h * dpr)
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
        init(n, w, h)
      }

      const edge = dark ? 'rgba(255,255,255,0.07)' : 'rgba(14,17,22,0.09)'
      const idle = dark ? '#3a4150' : '#aab0bb'

      ctx.clearRect(0, 0, w, h)

      for (const node of nodes) {
        node.x += node.vx
        node.y += node.vy
        if (node.x < 18 || node.x > w - 18) node.vx *= -1
        if (node.y < 18 || node.y > h - 18) node.vy *= -1
        if (node.pulse > 0) node.pulse -= 0.03
      }

      // edges
      ctx.lineWidth = 1
      ctx.strokeStyle = edge
      for (let i = 0; i < nodes.length; i++) {
        for (const j of adj[i]) {
          if (j <= i) continue
          ctx.beginPath()
          ctx.moveTo(nodes[i].x, nodes[i].y)
          ctx.lineTo(nodes[j].x, nodes[j].y)
          ctx.stroke()
        }
      }

      // pulses
      for (const pl of pulses) {
        pl.t += pl.speed * speed
        if (pl.t >= 1) {
          nodes[pl.b].pulse = 1
          pl.a = pl.b
          const nb = adj[pl.a]
          pl.b = nb.length ? nb[Math.floor(Math.random() * nb.length)] : pl.a
          pl.t = 0
        }
        const A = nodes[pl.a]
        const B = nodes[pl.b]
        const x = A.x + (B.x - A.x) * pl.t
        const y = A.y + (B.y - A.y) * pl.t
        ctx.strokeStyle = accent
        ctx.globalAlpha = 0.5
        ctx.lineWidth = 1.6
        ctx.beginPath()
        ctx.moveTo(A.x + (B.x - A.x) * Math.max(0, pl.t - 0.18), A.y + (B.y - A.y) * Math.max(0, pl.t - 0.18))
        ctx.lineTo(x, y)
        ctx.stroke()
        ctx.globalAlpha = 1
        ctx.fillStyle = accent
        if (glow) {
          ctx.shadowColor = accent
          ctx.shadowBlur = 10
        }
        if (shape === 'squares') {
          ctx.fillRect(x - 2.6, y - 2.6, 5.2, 5.2)
        } else {
          ctx.beginPath()
          ctx.arc(x, y, 2.6, 0, Math.PI * 2)
          ctx.fill()
        }
        ctx.shadowBlur = 0
      }

      // nodes
      for (const node of nodes) {
        const active = node.pulse > 0.05
        const col = active ? accent : idle
        if (node.pulse > 0) {
          ctx.globalAlpha = node.pulse * 0.4
          ctx.fillStyle = accent
          ctx.beginPath()
          ctx.arc(node.x, node.y, node.r + 7 * node.pulse, 0, Math.PI * 2)
          ctx.fill()
          ctx.globalAlpha = 1
        }
        if (shape === 'squares') {
          ctx.fillStyle = col
          const s = node.r * 1.7
          ctx.fillRect(node.x - s / 2, node.y - s / 2, s, s)
        } else if (shape === 'rings') {
          ctx.strokeStyle = col
          ctx.lineWidth = 1.5
          ctx.beginPath()
          ctx.arc(node.x, node.y, node.r + 0.6, 0, Math.PI * 2)
          ctx.stroke()
        } else {
          ctx.fillStyle = col
          ctx.beginPath()
          ctx.arc(node.x, node.y, node.r, 0, Math.PI * 2)
          ctx.fill()
        }
      }

      raf = requestAnimationFrame(draw)
    }

    raf = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(raf)
  }, [count, speed, shape, glow, accent, dark])

  return <canvas ref={canvasRef} style={style} />
}
