import { useRouter } from 'next/router'
import { useConfig } from 'nextra-theme-docs'

const description =
  'A decentralized, peer-to-peer LLM inference network — BOINC for the layers of an LLM.'

// The Eujeno mark — accent square with the 2x2 dot grid (same as the landing).
function LogoMark() {
  return (
    <svg width="22" height="22" viewBox="0 0 32 32" aria-hidden="true">
      <rect width="32" height="32" rx="7" fill="#4f46e5" />
      <g fill="#fff">
        <rect x="10" y="10" width="5" height="5" rx="1.2" />
        <rect x="17" y="10" width="5" height="5" rx="1.2" fillOpacity="0.5" />
        <rect x="10" y="17" width="5" height="5" rx="1.2" fillOpacity="0.5" />
        <rect x="17" y="17" width="5" height="5" rx="1.2" />
      </g>
    </svg>
  )
}

export default {
  logo: (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 9 }}>
      <LogoMark />
      <span style={{ fontWeight: 700, letterSpacing: '-0.01em' }}>
        Eujeno <span style={{ opacity: 0.55, fontWeight: 500 }}>Docs</span>
      </span>
    </span>
  ),
  project: {
    link: 'https://github.com/babelfornet/eujeno',
  },
  docsRepositoryBase: 'https://github.com/babelfornet/eujeno/tree/main/docs',
  color: {
    hue: 245,
    saturation: 80,
  },
  sidebar: {
    defaultMenuCollapseLevel: 1,
    toggleButton: true,
  },
  footer: {
    content: (
      <span>
        Apache-2.0 · <strong>Eujeno</strong> — peer-to-peer LLM inference network
      </span>
    ),
  },
  head: function Head() {
    const { basePath } = useRouter()
    const { title } = useConfig()
    const pageTitle = title ? `${title} – Eujeno Docs` : 'Eujeno Docs'
    return (
      <>
        <title>{pageTitle}</title>
        <link rel="icon" type="image/svg+xml" href={`${basePath}/favicon.svg`} />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <meta name="description" content={description} />
        <meta property="og:title" content={pageTitle} />
        <meta property="og:description" content={description} />
        <meta name="theme-color" content="#4f46e5" />
      </>
    )
  },
}
