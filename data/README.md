# Data

Tracked in git:

| Path | Contents |
|---|---|
| `antibacterial.fasta` | The organizers' exclusion set, 39,448 sequences, read at inference time for the novelty screen. sha256 `cbbeac64ba95746d87961e8ad9dd0849ae8058d15a300b2e7f6990730ca521e9`, identical to the file in szczurek-lab/amp-challenge-2027 (checked 2026-09-30) |
| `activity_sources.json` | Our specification of the MIC and HC50 sources: per-source column mappings and row filters. No third-party data |
| `processed/README.md` | Upstream sources, retrieval dates, rebuild command, expected row counts and checksum for the harmonized activity table |

Not tracked: `raw/` (downloads), `interim/` (scratch) and the contents of `processed/`.
The harmonized activity table is not redistributed. `processed/README.md` explains why and gives the rebuild path.
