# Rating pool through Cloudflare (runbook)

Replaces the ssh tunnels that workers and the report use today with two
Cloudflare Access-protected hostnames on mombox's existing tunnel (`fc-tunnel`,
tunnel `17931a78-a7b6-46e5-9e73-e7a5a9d61de5`, config `/etc/cloudflared/config.yml`,
mirrored in `~/.cloudflared/config.yml`). The coordinator's bearer token stays in
force; Access is a second layer in front of it.

Run the steps **in this order**. Access is created first, so the hostnames are
never reachable unprotected. The tunnel serves drmariostats: follow the
fightcadeRatings lockout-safety protocol (`ops/RUNBOOK.md`) and keep one working
session open while changing it.

## Hostnames: two, not one with path routing

| Hostname | Origin | Who | Access policy |
|---|---|---|---|
| `pool.zudark.net` | `http://192.168.157.190:8097` (pool API, bearer token) | workers, CLI, watchers | Service Auth (the pool service token), plus Allow your email for manual use |
| `poolreport.zudark.net` | `http://192.168.157.190:8098` (read-only report page) | you, in a browser | Allow your email |

Two hostnames because the two origins need different Access policies. Workers
authenticate non-interactively with a service token. The report is for a person
behind an email login, and the report server itself has no authentication.
Path routing on one hostname would put both origins under one Access application
and one policy. It would also depend on a path split between `/api/…` and `/` that
the report page (`/report.json`) blurs. Separate hosts also keep the report's
browser session cookies away from the worker API.

## 1. Cloudflare Access first (no DNS yet)

### (a) Dashboard

1. Zero Trust → Access → Service Auth → Service Tokens → **Create Service Token**:
   name `drmc-rl-pool-workers`, duration 1 year. Copy the Client ID and Client
   Secret now; the secret is shown once. Store them only in the env file of step 6.
2. Zero Trust → Access → Applications → **Add an application** → Self-hosted:
   - Name `drmc-rl pool API`; application domain `pool.zudark.net`; session 24 h;
     hide it from the App Launcher.
   - Policy 1: name `workers`, Action **Service Auth**, Include → Service Token →
     `drmc-rl-pool-workers`.
   - Policy 2: name `owner`, Action **Allow**, Include → Emails → your address.
3. **Add an application** again → Self-hosted: name `drmc-rl pool report`,
   domain `poolreport.zudark.net`, session 24 h, one policy `owner` (Allow, your
   email). Optionally add the service-token Service Auth policy too, if you want
   scripts to fetch `report.json`.

### (b) API (you run these; the API token never goes to an agent)

Create an API token with **Access: Apps and Policies — Edit** and **Access:
Service Tokens — Edit** for the account. Then:

```bash
export CF_ACCOUNT_ID=<account id>            # dashboard → any zone → Overview → Account ID
export CF_API_TOKEN=<api token>
export OWNER_EMAIL=<your email>
api() { curl -sS -H "Authorization: Bearer $CF_API_TOKEN" -H "Content-Type: application/json" "$@"; }
A=https://api.cloudflare.com/client/v4/accounts/$CF_ACCOUNT_ID/access

# Service token (the response holds client_id and client_secret; the secret is shown once)
api -X POST $A/service_tokens -d '{"name":"drmc-rl-pool-workers","duration":"8760h"}' > /tmp/pool-token.json
TOKEN_ID=$(jq -r .result.id /tmp/pool-token.json)
jq -r '"CF_ACCESS_CLIENT_ID=\(.result.client_id)\nCF_ACCESS_CLIENT_SECRET=\(.result.client_secret)"' \
  /tmp/pool-token.json > ~/pool-access.env && chmod 600 ~/pool-access.env && shred -u /tmp/pool-token.json

# API application and its policies
APP=$(api -X POST $A/apps -d '{"name":"drmc-rl pool API","domain":"pool.zudark.net","type":"self_hosted",
  "session_duration":"24h","app_launcher_visible":false}' | jq -r .result.id)
api -X POST $A/apps/$APP/policies -d "{\"name\":\"workers\",\"decision\":\"non_identity\",\"precedence\":1,
  \"include\":[{\"service_token\":{\"token_id\":\"$TOKEN_ID\"}}]}"
api -X POST $A/apps/$APP/policies -d "{\"name\":\"owner\",\"decision\":\"allow\",\"precedence\":2,
  \"include\":[{\"email\":{\"email\":\"$OWNER_EMAIL\"}}]}"

# Report application
REP=$(api -X POST $A/apps -d '{"name":"drmc-rl pool report","domain":"poolreport.zudark.net","type":"self_hosted",
  "session_duration":"24h","app_launcher_visible":false}' | jq -r .result.id)
api -X POST $A/apps/$REP/policies -d "{\"name\":\"owner\",\"decision\":\"allow\",\"precedence\":1,
  \"include\":[{\"email\":{\"email\":\"$OWNER_EMAIL\"}}]}"

api $A/apps | jq '.result[] | {name, domain, id}'        # check both applications exist
```

(If your account only accepts reusable policies, create them under
`$A/policies` and attach them with `"policies":[{"id":...}]` on the application.)

## 2. Tunnel ingress (mombox)

Add these two rules **before** the final `http_status:404` rule in
`/etc/cloudflared/config.yml`, and add the same rules to the mirror in
`~/.cloudflared/config.yml`:

```yaml
  - hostname: pool.zudark.net
    service: http://192.168.157.190:8097
  - hostname: poolreport.zudark.net
    service: http://192.168.157.190:8098
```

The coordinator binds the LAN address only (not localhost), so the origin is
`192.168.157.190`. Check the file before restarting anything:

```bash
sudo cloudflared --config /etc/cloudflared/config.yml tunnel ingress validate
sudo cloudflared --config /etc/cloudflared/config.yml tunnel ingress rule https://pool.zudark.net
sudo cloudflared --config /etc/cloudflared/config.yml tunnel ingress rule https://poolreport.zudark.net
sudo cloudflared --config /etc/cloudflared/config.yml tunnel ingress rule https://drmariostats.zudark.net   # unchanged
```

## 3. DNS

```bash
cloudflared tunnel route dns 17931a78-a7b6-46e5-9e73-e7a5a9d61de5 pool.zudark.net
cloudflared tunnel route dns 17931a78-a7b6-46e5-9e73-e7a5a9d61de5 poolreport.zudark.net
```

(Run as `ethan`; `~/.cloudflared/cert.pem` authorises it.)

## 4. Restart the tunnel

```bash
sudo systemctl restart fc-tunnel         # drmariostats blips for a few seconds
systemctl status fc-tunnel --no-pager | head -5
journalctl -u fc-tunnel -n 30 --no-pager  # expect four "Registered tunnel connection" lines
curl -s -o /dev/null -w '%{http_code} %{time_total}\n' https://drmariostats.zudark.net/
curl -s -o /dev/null -w '%{time_total}\n' http://localhost:8762/
```

## 5. Caching

The pool API sends `Cache-Control: no-store`, and the report sends `no-cache`
(page) and `no-store` (`report.json`). Checkpoint downloads
(`/api/v1/checkpoints/<sha>`) have no file extension, so Cloudflare's default
cache does not store them. A bypass rule is optional, as belt and braces:
Caching → Cache Rules → Create rule `pool bypass`, expression
`(http.host in {"pool.zudark.net" "poolreport.zudark.net"})`, cache eligibility
**Bypass cache**. Or by API, with a token that has Zone → Cache Rules — Edit:

```bash
ZONE=<zudark.net zone id>
curl -sS -X PUT -H "Authorization: Bearer $CF_API_TOKEN" -H "Content-Type: application/json" \
  https://api.cloudflare.com/client/v4/zones/$ZONE/rulesets/phases/http_request_cache_settings/entrypoint \
  -d '{"rules":[{"expression":"(http.host in {\"pool.zudark.net\" \"poolreport.zudark.net\"})",
       "action":"set_cache_settings","action_parameters":{"cache":false},"description":"pool bypass"}]}'
```

(This PUT replaces the zone's cache rules. If any exist, add the rule in the
dashboard instead.)

## 6. Switch the workers

On every worker host (Mac, green, tf3090):

```bash
install -m 600 /dev/null ~/.config/drmc-rl/pool-access.env
# put the two lines from step 1 into it:
#   CF_ACCESS_CLIENT_ID=...
#   CF_ACCESS_CLIENT_SECRET=...
export DRMC_POOL_URL=https://pool.zudark.net
```

The pool client reads the file automatically (it refuses a file readable by
others). Mac: set `DRMC_POOL_URL=https://pool.zudark.net` at the top of
`drmc-rl-pool-data/start-mac-workers.sh` and `start-watchers.sh`. Restart the
workers with `pkill -TERM -f "tools.rating_pool worker"` (leases are released)
and then run the start scripts. Stop the ssh tunnel:
`pkill -f drmc-rl-pool-data/tunnel.sh; pkill -f "ssh -N .*8097"`. Green and
tf3090 use the same URL with `--coordinator https://pool.zudark.net` and need
no LAN route, ssh key or firewall rule. The report is at
https://poolreport.zudark.net/ after an email login.

Checkpoint registration from any host (`entrant add`, `bootstrap`,
`watch-run`) uploads in 32 MB parts plus a final sha256 check, below
Cloudflare's 100 MB request-body limit. Downloads stream through Cloudflare, and
the coordinator's limits of 2 streams at 20 MB/s each still apply.

## 7. Verify

```bash
T=$(cat ~/.config/drmc-rl/study-worker.token); set -a; . ~/.config/drmc-rl/pool-access.env; set +a
# no service token: Access refuses (302 to the login page, or 403)
curl -s -o /dev/null -w '%{http_code} %{redirect_url}\n' https://pool.zudark.net/api/v1/pool/study
# service token, no bearer: the coordinator refuses (401)
curl -s -o /dev/null -w '%{http_code}\n' -H "CF-Access-Client-Id: $CF_ACCESS_CLIENT_ID" \
  -H "CF-Access-Client-Secret: $CF_ACCESS_CLIENT_SECRET" https://pool.zudark.net/api/v1/pool/study
# both: 200 JSON
curl -s -H "CF-Access-Client-Id: $CF_ACCESS_CLIENT_ID" -H "CF-Access-Client-Secret: $CF_ACCESS_CLIENT_SECRET" \
  -H "Authorization: Bearer $T" https://pool.zudark.net/api/v1/pool/study | jq .protocol
# the report host without a login: 302 to Access
curl -s -o /dev/null -w '%{http_code}\n' https://poolreport.zudark.net/
# a real lease through Cloudflare (from the pool checkout)
DRMC_POOL_URL=https://pool.zudark.net python -m tools.rating_pool summary | head -5
DRMC_POOL_URL=https://pool.zudark.net python -m tools.rating_pool worker --device mps --max-batches 1 \
  --native-library $N/libdrmario_pool.dylib --reach-library $N/libdrm_reach_full.dylib
```

A client without the token fails with a clear `AccessDenied` message naming
`~/.config/drmc-rl/pool-access.env`, not with a JSON parse error.

## 8. Rollback

1. Workers: `export DRMC_POOL_URL=http://127.0.0.1:8097`, restart the ssh
   tunnel (`nohup drmc-rl-pool-data/tunnel.sh &`) and restart the workers.
2. Remove the two ingress rules from `/etc/cloudflared/config.yml` and the
   mirror, run `tunnel ingress validate`, then `sudo systemctl restart fc-tunnel`,
   and check drmariostats as in step 4.
3. DNS: delete the `pool` and `poolreport` CNAME records (dashboard → zudark.net
   → DNS). `cloudflared tunnel route dns` does not delete.
4. Access: delete or keep the two applications. Revoke the service token (Zero
   Trust → Access → Service Auth) if it may have leaked, and delete
   `~/.config/drmc-rl/pool-access.env` on each host.
