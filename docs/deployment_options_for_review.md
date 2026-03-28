# Deployment Options — Expert Review

**Context:** Single-user NGO wildlife labeling app. Rangers label trail camera photos remotely in a browser (no local install). All ML inference stays on a local M3 Mac. The hosted system only needs to show images and save label assignments (deer ID + side per image). ~660 JPEGs, one active user at a time.

---

## Option A — AWS S3 + Lambda (Python)

**What it is:** Images and labels stored in Amazon S3. A small Python function (Lambda) handles the API. Frontend is a static HTML/JS page.

**Cost:**
- Storage: ~$0.07/month (3GB thumbnails)
- Idle months: ~$0.07
- Active months: ~$0.30–0.60 (includes image browsing egress)
- ~$3–5/year at moderate use

**Pros:**
- Pay only when used — idle months cost almost nothing
- Python throughout
- Best-documented cloud platform — lowest AI agent error rate
- No server to manage

**Cons:**
- Image egress charges add up if full-resolution photos are served (mitigated by thumbnails)
- More setup than a VPS: IAM roles, Lambda packaging, S3 bucket policy, CORS
- Lambda Function URLs needed to avoid extra API Gateway cost

---

## Option B — Hetzner VPS (€3.29/month)

**What it is:** A small Linux server in Germany. Run the existing Python web server directly on it, behind nginx. Deploy via git pull.

**Cost:**
- €3.29/month flat (~$43/year)
- No egress charges (20TB/month included)
- Same cost whether used or idle

**Pros:**
- Minimal code change — existing Python server deploys almost as-is
- No vendor-specific APIs to learn
- Best AI agent reliability (standard Linux + Python)
- Predictable cost

**Cons:**
- Always-on cost even in idle months
- You manage the server: OS patching, SSL renewal, uptime monitoring
- If it goes down at 2am, nobody restarts it automatically (UptimeRobot free tier covers alerting)

---

## Option C — Oracle Cloud Always Free

**What it is:** Oracle's permanent free tier. 4 ARM vCPUs, 24GB RAM, 200GB disk, 10TB bandwidth/month. Run the Python server on it, same as Option B but at $0.

**Cost:**
- $0/month, no expiry

**Pros:**
- Genuinely free forever
- Same Python/Linux environment as Hetzner
- More resources than needed — no capacity concerns

**Cons:**
- Oracle Cloud's console and documentation are significantly worse than AWS or Hetzner
- AI agent reliability is lower — fewer tutorials, less community support
- Account setup requires a credit card (won't be charged, but required)
- Oracle's reputation for aggressive sales and surprise billing changes makes some teams uncomfortable

---

## Option D — Cloudflare Pages + R2 + Workers

**What it is:** Static frontend on Cloudflare Pages, images in R2 object storage (zero egress fees), API logic in Cloudflare Workers.

**Cost:**
- R2 storage: ~$0.015/month for 3GB thumbnails
- Workers + Pages: free tier covers this workload
- ~$0.05–0.20/month

**Pros:**
- Cheapest pay-per-use option
- Zero image egress fees (R2's key advantage over S3)
- No server to manage

**Cons:**
- **All backend API logic must be rewritten in JavaScript** — the existing Python codebase cannot run on Workers
- Workers KV (label storage) is eventually consistent: a ranger may not see their own label update for up to 60 seconds after saving. Fix costs +$5/month (Durable Objects).
- Smaller ecosystem — AI agent error rate is higher, development costs more in tokens and time
- Any future backend features stay locked to JavaScript

---

## Summary

| | Monthly cost | Idle cost | Server mgmt | Python | Agent reliability |
|---|---|---|---|---|---|
| A — AWS Lambda | ~$0.30–0.60 active, ~$0.07 idle | ✓ low | None | ✓ | High |
| B — Hetzner VPS | €3.29 flat | ✗ same | Moderate | ✓ | High |
| C — Oracle Free | $0 | ✓ free | Moderate | ✓ | Medium |
| D — Cloudflare | ~$0.05–0.20 | ✓ low | None | ✗ JS rewrite | Medium |

**Key question for the expert:** How many months per year will rangers actively use the labeling UI?
- If 1–3 months/year → **Option A or D** (pay-per-use wins)
- If 6+ months/year → **Option B** (flat rate becomes competitive, simplest ops)
- If cost must be $0 and JS rewrite is acceptable → **Option D**
- If cost must be $0 and Python matters → **Option C** (accepting Oracle's DX trade-off)
