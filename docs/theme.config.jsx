import { useConfig } from 'nextra-theme-docs'

const description =
  'A decentralized, peer-to-peer LLM inference network — BOINC for the layers of an LLM.'

export default {
  logo: (
    <span style={{ fontWeight: 700, letterSpacing: '-0.01em' }}>
      Eujeno <span style={{ opacity: 0.55, fontWeight: 500 }}>Docs</span>
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
    const { title } = useConfig()
    const pageTitle = title ? `${title} – Eujeno Docs` : 'Eujeno Docs'
    return (
      <>
        <title>{pageTitle}</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <meta name="description" content={description} />
        <meta property="og:title" content={pageTitle} />
        <meta property="og:description" content={description} />
        <meta name="theme-color" content="#4f46e5" />
      </>
    )
  },
}
