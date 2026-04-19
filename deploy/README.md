# Deploy recipes

Single-machine deployment targets for the Sentinel scheduler daemon. All
recipes reuse the root [`Dockerfile`](../Dockerfile) — nothing here is
platform-specific to Sentinel's code.

## Fly.io (`fly.toml`)

Small-footprint scheduler daemon on DuckDB with a persistent volume.
Good default for a personal research deployment; scales to Postgres by
attaching a Fly Postgres cluster and flipping two env vars.

```bash
# From the repo root, one-time:
fly launch --copy-config --dockerfile Dockerfile --no-deploy
fly volumes create sentinel_data --region iad --size 3

fly secrets set \
    REDDIT_CLIENT_ID=... \
    REDDIT_CLIENT_SECRET=... \
    REDDIT_USER_AGENT="sentinel/0.1 by <you>" \
    TWITTER_BEARER_TOKEN=...

fly deploy --config deploy/fly.toml
```

The container keeps running under `tini` + Fly's process manager; the
`sentinel version` healthcheck recycles the VM if the import graph
breaks (e.g., an optional dep that failed to resolve on the platform).

State lives on the named volume `sentinel_data` mounted at `/data` —
survives deploys, survives VM swaps. The DuckDB file, MLflow local
runs, and any CLI-generated reports all land there.

### Upgrading to Postgres on Fly

1. `fly pg create` — standard Fly Postgres cluster.
2. `fly pg attach <cluster> -a sentinel` — injects `DATABASE_URL`.
3. Set secrets:
   ```bash
   fly secrets set \
       SENTINEL_STORAGE_BACKEND=postgres \
       SENTINEL_POSTGRES_DSN='$DATABASE_URL' \
       SENTINEL_POSTGRES_TIMESCALE=false
   ```
4. Redeploy — the container's schema-bootstrap runs on first connect.

(Fly Postgres does not ship the Timescale extension. Sentinel's schema
soft-falls-back to plain Postgres tables, so this works without any
code change.)

## Other platforms

The `docker-compose.yml` at the repo root is the reference for
multi-service deployments (Render Blueprints, Railway,
Docker-on-a-VPS). The three profiles — default, `postgres`, `mlflow`
— map cleanly onto each platform's service model, since they already
declare explicit volumes, healthchecks, and inter-service dependencies.
