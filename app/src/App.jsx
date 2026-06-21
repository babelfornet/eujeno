import React, { useState, useEffect } from 'react'
import { THEMES, ACCENTS, loadThemePrefs, saveThemePrefs } from './theme.js'
import { getNode, getMetrics } from './api.js'
import Sidebar from './Sidebar.jsx'
import NetworkPage from './NetworkPage.jsx'
import ChatPage from './ChatPage.jsx'

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
          <NetworkPage T={T} accent={accent} dark={theme === 'dark'} node={node} metrics={metrics} />
        )}
        {view === 'chat' && (
          <ChatPage T={T} accent={accent} />
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
