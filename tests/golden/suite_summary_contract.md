# Research Suite Summary

- Generated (UTC): `<timestamp>`
- Env: `CartPole-v1`
- Episodes: `8`
- Seed: `7`
- Speeds: `[1, 2, 5]`
- Resume: `True`

| Stage | Status | Deployment | Stress | Reason |
| --- | --- | --- | --- | --- |
| deliberative | success | 0.910 | 0.640 |  |
| ltc | cached | 0.880 | 0.580 | existing summary.json reused (--resume) |
| bridge | skipped | n/a | n/a | Bridge stage needs a successful deliberative or ltc stage. |

## Recommendations

1. Resolve skipped stages (bridge) by adding prerequisites or choosing a compatible env/action space.
