import React from 'react'

// Icons
function IconNetwork() {
  return (
    <svg style={{flex:'none'}} width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="5" cy="6" r="2.2"/><circle cx="19" cy="6" r="2.2"/><circle cx="12" cy="18" r="2.2"/>
      <path d="M6.8 7.2 10.6 16M17.2 7.2 13.4 16M7 6h10"/>
    </svg>
  )
}

function IconChat() {
  return (
    <svg style={{flex:'none'}} width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 5.5h16v10.5H9l-4 3.5v-3.5H4z"/>
      <path d="M8.5 10.5h7M8.5 13h4.5"/>
    </svg>
  )
}

function IconSettings() {
  return (
    <svg style={{flex:'none'}} width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="3.1"/>
      <path d="M12 3.5v2.3M12 18.2v2.3M3.5 12h2.3M18.2 12h2.3M6 6l1.6 1.6M16.4 16.4 18 18M18 6l-1.6 1.6M7.6 16.4 6 18"/>
    </svg>
  )
}

function IconChevronLeft() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M15 6l-6 6 6 6"/>
    </svg>
  )
}

// 4-dot logo grid (2x2)
function LogoSquare({ accent }) {
  return (
    <span style={{flex:'none', width:'26px', height:'26px', borderRadius:'7px', background:accent, display:'grid', placeItems:'center'}}>
      <span style={{display:'grid', gridTemplateColumns:'repeat(2,4.5px)', gap:'2.5px'}}>
        <span style={{width:'4.5px',height:'4.5px',background:'#fff',borderRadius:'1px'}}/>
        <span style={{width:'4.5px',height:'4.5px',background:'rgba(255,255,255,0.5)',borderRadius:'1px'}}/>
        <span style={{width:'4.5px',height:'4.5px',background:'rgba(255,255,255,0.5)',borderRadius:'1px'}}/>
        <span style={{width:'4.5px',height:'4.5px',background:'#fff',borderRadius:'1px'}}/>
      </span>
    </span>
  )
}

export default function Sidebar({ view, setView, collapsed, toggle, T, accent, node, metrics }) {
  const asideW     = collapsed ? '70px'        : '248px'
  const asidePad   = collapsed ? '18px 10px'   : '18px 14px'
  const navJustify = collapsed ? 'center'       : 'flex-start'
  const navPad     = collapsed ? '11px 0'       : '10px 12px'
  const labelDisp  = collapsed ? 'none'         : 'inline'
  const statusPad  = collapsed ? '12px 0'       : '13px'

  const navItem = (id) => ({
    bg:    view === id ? `color-mix(in srgb, ${accent} 13%, transparent)` : 'transparent',
    color: view === id ? accent : T.muted,
    bar:   view === id ? '1' : '0',
  })
  const navNet  = navItem('network')
  const navChat = navItem('chat')
  const navSet  = navItem('settings')

  // Status card values
  const peerId  = node?.peerId  ? `node·${node.peerId.slice(0,6)}`  : 'node·—'
  const peerCnt = metrics?.connectedPeers ?? 0
  const model   = node?.model    ?? '—'

  return (
    <aside style={{
      flex: `0 0 ${asideW}`, width: asideW, minWidth: 0, overflow: 'hidden',
      borderRight: `1px solid ${T.border}`, background: T.section,
      display: 'flex', flexDirection: 'column', padding: asidePad,
      transition: 'flex-basis 0.18s ease, width 0.18s ease, padding 0.18s ease',
    }}>

      {/* Logo row */}
      <div style={{display:'flex', alignItems:'center', gap:'8px', padding:'6px 4px 18px'}}>
        <button
          onClick={toggle}
          title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          style={{flex:'1', minWidth:0, display:'flex', alignItems:'center', justifyContent:navJustify,
                  gap:'10px', border:'none', background:'transparent', cursor:'pointer', padding:'0'}}
        >
          <LogoSquare accent={accent} />
          <span style={{flex:'1', textAlign:'left', color:T.text, fontWeight:'700', fontSize:'18px',
                        letterSpacing:'-0.01em', whiteSpace:'nowrap', overflow:'hidden', display:labelDisp}}>
            Eujeno
          </span>
        </button>

        {/* Collapse chevron button (hidden when collapsed) */}
        {!collapsed && (
          <button
            onClick={toggle}
            title="Collapse sidebar"
            style={{flex:'none', display:'grid', placeItems:'center', width:'28px', height:'28px',
                    borderRadius:'7px', border:`1px solid ${T.borderStrong}`, background:T.cardBg,
                    color:T.muted, cursor:'pointer'}}
          >
            <IconChevronLeft />
          </button>
        )}
      </div>

      {/* Nav */}
      <nav style={{display:'flex', flexDirection:'column', gap:'3px'}}>
        {[
          { id: 'network',  label: 'Network',  Icon: IconNetwork,  nav: navNet  },
          { id: 'chat',     label: 'Chat',     Icon: IconChat,     nav: navChat },
          { id: 'settings', label: 'Settings', Icon: IconSettings, nav: navSet  },
        ].map(({ id, label, Icon, nav }) => (
          <button
            key={id}
            onClick={() => setView(id)}
            title={label}
            style={{
              position:'relative', display:'flex', alignItems:'center', justifyContent:navJustify,
              gap:'11px', width:'100%', textAlign:'left', border:'none', cursor:'pointer',
              padding:navPad, borderRadius:'9px', fontSize:'14.5px', fontWeight:'600',
              background: nav.bg, color: nav.color,
            }}
          >
            {/* Active accent bar */}
            <span style={{
              position:'absolute', left:'-10px', top:'50%', transform:'translateY(-50%)',
              width:'3px', height:'20px', borderRadius:'0 3px 3px 0',
              background: accent, opacity: nav.bar,
            }}/>
            <Icon />
            <span style={{display:labelDisp, whiteSpace:'nowrap'}}>{label}</span>
          </button>
        ))}
      </nav>

      <div style={{flex:'1'}}/>

      {/* Status card */}
      <div style={{border:`1px solid ${T.border}`, background:T.cardBg, borderRadius:'12px', padding:statusPad}}>
        <div style={{display:'flex', alignItems:'center', gap:'8px', justifyContent:navJustify}}>
          <span style={{flex:'none', width:'8px', height:'8px', borderRadius:'50%',
                        background:'#16a34a', boxShadow:'0 0 0 3px rgba(22,163,74,0.16)'}}/>
          <span style={{fontSize:'13px', fontWeight:'600', color:T.text, display:labelDisp}}>Connected</span>
        </div>
        <div style={{marginTop:'9px', fontFamily:"'JetBrains Mono',monospace", fontSize:'11px',
                     color:T.muted2, lineHeight:'1.6', display:labelDisp}}>
          <div>{peerId}</div>
          <div>{peerCnt} peers · {model}</div>
        </div>
      </div>
    </aside>
  )
}
