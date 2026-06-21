const J = (r) => r.ok ? r.json() : Promise.reject(new Error(r.status))
export const getNode    = () => fetch('/api/node').then(J)
export const getMetrics = () => fetch('/api/metrics').then(J)
export const getPeers   = () => fetch('/api/peers').then(J)
export const getSettings= () => fetch('/api/settings').then(J)
export const putSettings= (b) => fetch('/api/settings', {method:'PUT', headers:{'content-type':'application/json'}, body:JSON.stringify(b)}).then(J)
export const restart    = () => fetch('/api/node/restart', {method:'POST'}).then(J)
export const chat = (messages, max_tokens=256) =>
  fetch('/v1/chat/completions', {method:'POST', headers:{'content-type':'application/json'},
    body:JSON.stringify({messages, max_tokens})}).then(J)
