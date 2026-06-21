import nextra from 'nextra'

const withNextra = nextra({
  theme: 'nextra-theme-docs',
  themeConfig: './theme.config.jsx',
  defaultShowCopyCode: true,
})

// Static export for GitHub Pages. basePath is absolute (unlike Vite's relative
// base) so it must include the full path from the domain root. The CI sets
// BASE_PATH=/eujeno/docs; for a future apex domain (eujeno.com) use /docs.
export default withNextra({
  output: 'export',
  images: { unoptimized: true },
  basePath: process.env.BASE_PATH || '',
  trailingSlash: true,
})
