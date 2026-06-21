# Eujeno Node Dashboard

React + Vite app that provides the per-node UI (Network / Chat / Settings tabs).

## Development

```bash
npm install
npm run dev        # dev server at http://localhost:5173 (needs a running node — see below)
```

For the dev server to reach the node's `/api` and `/v1` endpoints, either:
- point the Vite proxy at a running node (see `vite.config.js`), or
- serve the built bundle from the node itself (`npm run build` first, then `eujeno serve ...`).

## Building

```bash
npm run build
```

Output goes to `../eujeno/ui/static/` (configured in `vite.config.js` with `base: './'`).
The built files are committed to the repo so nodes serve the dashboard with zero npm
dependency at runtime — `eujeno serve` just picks them up from `eujeno/ui/static/`.

## How nodes serve it

`eujeno/net/server.py` (`create_app`) mounts the static directory at `/`. Every node
running `eujeno serve` exposes the dashboard at its own URL (`http://<host>:<port>/`).

Open it with:

```bash
eujeno ui --node http://127.0.0.1:8001
```
