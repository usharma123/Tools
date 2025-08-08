## Benched Web – Next.js UI

Clean, modern UI for Benched with a centered hero, pill input, and continuous purple gradients.

### Dev

```bash
pnpm install
pnpm dev
```

### Build

```bash
pnpm build
pnpm start
```

### Notes
- Title updated to "Benched" in `src/app/layout.tsx` metadata.
- Global gradients and aurora in `src/app/globals.css` with noise overlay to avoid banding.
- Hero and input in `src/app/page.tsx`. Section gradients render only when results exist.
