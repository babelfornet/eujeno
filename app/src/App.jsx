import React, { useState, useEffect } from 'react'
import { THEMES, ACCENTS, loadThemePrefs, saveThemePrefs } from './theme.js'
import { getNode, getMetrics } from './api.js'
import Sidebar from './Sidebar.jsx'

export default function App() {
  // Theme state
  const initial = loadThemePrefs()
  const [theme,  setThemeState]  = useState(initial.theme)
  const [accent, setAccentState] = useState(initial.accent)

  function setTheme(t)  { setThemeState(t);  saveThemePrefs(t, accent) }
  function setAccent(a) { setAccentState(a); saveThemePrefs(theme, a)  }

  const T = THEMES[theme]

  // View + sidebar state
  const [view,      setView]      = useState('network')
  const [collapsed, setCollapsed] = useState(false)

  // Live node / metrics data
  const [node,    setNode]    = useState(null)
  const [metrics, setMetrics] = useState(null)

  useEffect(() => {
    let cancelled = false

    async function poll() {
      try {
        const [n, m] = await Promise.all([getNode(), getMetrics()])
        if (!cancelled) { setNode(n); setMetrics(m) }
      } catch (_) { /* best-effort */ }
    }

    poll()
    const id = setInterval(poll, 2500)
    return () => { cancelled = true; clearInterval(id) }
  }, [])

  // CSS variables applied on root container — mirrors comp line 26
  const cssVars = {
    '--accent':        accent,
    '--page-bg':       T.pageBg,
    '--card-bg':       T.cardBg,
    '--elev-bg':       T.elevBg,
    '--text':          T.text,
    '--muted':         T.muted,
    '--muted2':        T.muted2,
    '--border':        T.border,
    '--border-strong': T.borderStrong,
    '--section-bg':    T.section,
    '--canvas-bg':     T.canvasBg,
    '--shadow':        T.shadow,
  }

  return (
    <div style={{
      ...cssVars,
      height: '100vh',
      display: 'flex',
      background: T.pageBg,
      color: T.text,
      fontFamily: "'Hanken Grotesk',system-ui,sans-serif",
      WebkitFontSmoothing: 'antialiased',
      overflow: 'hidden',
    }}>
      <Sidebar
        view={view}
        setView={setView}
        collapsed={collapsed}
        toggle={() => setCollapsed(c => !c)}
        T={T}
        accent={accent}
        node={node}
        metrics={metrics}
      />

      <main style={{flex:'1', minWidth:0, height:'100vh', display:'flex', flexDirection:'column', overflow:'hidden'}}>
        {view === 'network' && (
          <div style={{flex:'1', display:'flex', alignItems:'center', justifyContent:'center',
                       flexDirection:'column', gap:'8px', color:T.muted}}>
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="5" cy="6" r="2.2"/><circle cx="19" cy="6" r="2.2"/><circle cx="12" cy="18" r="2.2"/>
              <path d="M6.8 7.2 10.6 16M17.2 7.2 13.4 16M7 6h10"/>
            </svg>
            <span style={{fontSize:'15px', fontWeight:'600'}}>Network</span>
            <span style={{fontSize:'13px', color:T.muted2}}>Coming soon</span>
          </div>
        )}
        {view === 'chat' && (
          <div style={{flex:'1', display:'flex', alignItems:'center', justifyContent:'center',
                       flexDirection:'column', gap:'8px', color:T.muted}}>
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M4 5.5h16v10.5H9l-4 3.5v-3.5H4z"/>
              <path d="M8.5 10.5h7M8.5 13h4.5"/>
            </svg>
            <span style={{fontSize:'15px', fontWeight:'600'}}>Chat</span>
            <span style={{fontSize:'13px', color:T.muted2}}>Coming soon</span>
          </div>
        )}
        {view === 'settings' && (
          <div style={{flex:'1', display:'flex', alignItems:'center', justifyContent:'center',
                       flexDirection:'column', gap:'8px', color:T.muted}}>
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="3.1"/>
              <path d="M12 3.5v2.3M12 18.2v2.3M3.5 12h2.3M18.2 12h2.3M6 6l1.6 1.6M16.4 16.4 18 18M18 6l-1.6 1.6M7.6 16.4 6 18"/>
            </svg>
            <span style={{fontSize:'15px', fontWeight:'600'}}>Settings</span>
            <span style={{fontSize:'13px', color:T.muted2}}>Coming soon</span>
          </div>
        )}
      </main>
    </div>
  )
}
