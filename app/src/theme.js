// Light/dark token tables ported from the design comp (renderVals, lines 497–503)
export const THEMES = {
  light: {
    pageBg:      '#ffffff',
    cardBg:      '#ffffff',
    elevBg:      '#ffffff',
    text:        '#0e1116',
    muted:       '#5b6471',
    muted2:      '#7a828e',
    border:      '#e9ebef',
    borderStrong:'#d9dce2',
    section:     '#f7f8fa',
    canvasBg:    '#fafbfc',
    shadow:      '0 1px 2px rgba(16,24,40,0.05)',
  },
  dark: {
    pageBg:      '#0b0d12',
    cardBg:      '#14171f',
    elevBg:      '#1a1e28',
    text:        '#f2f4f7',
    muted:       '#9aa3b2',
    muted2:      '#7a8494',
    border:      '#20242e',
    borderStrong:'#2a2f3b',
    section:     '#0f1217',
    canvasBg:    '#0e1117',
    shadow:      'none',
  },
}

// Accent palette from comp line 325 (accentColor options)
export const ACCENTS = [
  '#4f46e5', // indigo (default)
  '#2563eb', // blue
  '#0d9488', // teal
  '#7c3aed', // violet
  '#e8590c', // orange
  '#e11d48', // rose
]

const STORAGE_KEY = 'eujeno_theme'

export function loadThemePrefs() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (raw) {
      const parsed = JSON.parse(raw)
      return {
        theme:  (parsed.theme  === 'dark' ? 'dark' : 'light'),
        accent: ACCENTS.includes(parsed.accent) ? parsed.accent : ACCENTS[0],
      }
    }
  } catch (_) {}
  return { theme: 'light', accent: ACCENTS[0] }
}

export function saveThemePrefs(theme, accent) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ theme, accent }))
  } catch (_) {}
}
