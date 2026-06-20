# Eujeno — landing site

Static **React (Vite)** build of the Eujeno landing page, ported from the
DivMagic design template in [`template/`](./template/). Outputs plain
HTML/CSS/JS to `dist/` — no server, hostable on GitHub Pages (or any static host).

## Develop

```bash
cd web
npm install
npm run dev        # http://localhost:5173
```

## Build

```bash
npm run build      # -> web/dist (static, self-contained)
npm run preview    # serve the production build locally
```

## Deploy (GitHub Pages)

The site auto-deploys via [`.github/workflows/deploy.yml`](../.github/workflows/deploy.yml)
on every push to `main` that touches `web/**`.

One-time setup: **repo Settings → Pages → Build and deployment → Source = GitHub Actions**.

`vite.config.js` uses `base: './'` (relative asset paths), so the same build works
both at a project path (`https://<user>.github.io/eujeno/`) and at the apex domain.

### Custom domain (eujeno.com)

To serve it at `eujeno.com`, add a `CNAME` file so Pages keeps the domain across deploys:

```bash
echo 'eujeno.com' > web/public/CNAME
```

Then point DNS at GitHub Pages (apex `A`/`AAAA` records or a `CNAME` to
`<user>.github.io`) and set the custom domain under Settings → Pages.

## Structure

```
web/
  index.html              # entry document (fonts, meta)
  vite.config.js          # base:'./', react plugin
  src/
    main.jsx              # React root
    App.jsx               # composes all sections
    theme.js              # template defaults as CSS variables
    styles.js             # shared style fragments
    index.css             # reset, keyframes, hovers, responsive
    components/
      Nav, Hero, SwarmCanvas, HowItWorks,
      WhyP2P, RunNode, UseCases, CTA, Footer
  template/               # original DivMagic export (design reference)
```

`SwarmCanvas.jsx` is a faithful port of the template's animated `<canvas>` swarm
(drifting peer nodes with token pulses along the edges).
