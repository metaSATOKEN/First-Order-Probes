# Manuscript distribution check

The current editions are [English V1.0](../Calibration_Update_Audits_for_First_Order_Probes_EN_v1.0.md) and [Japanese V1.0](../Calibration_Update_Audits_for_First_Order_Probes_JA_v1.0.md), dated 30 October 2026. [Version history](../docs/VERSION_HISTORY.md) maps their previous internal draft labels to this shared edition.

The V1.0 check compares each manuscript with its predecessor in commit `be3ec0dd2db2851dbfbe559748c9f523bfac6237`. Reversing the version/date header change restores the preceding file's complete bytes. The scientific prose, 24 displayed equations, tables, figure references, and bibliography are therefore preserved. Source/distribution hashes and the transformations are recorded in [source_manifest.json](source_manifest.json); the version-transition checks are in [V1_0_CHECK.json](V1_0_CHECK.json).

Earlier checks have separate dates and scopes. The [original distribution summary](MANUSCRIPT_CHECK_2026-10-27.md) records the preceding manuscript audit, including 17 table bodies, 11 figures per language, 15 references, and 140 checked numerical displays. The [local-corrections record](LOCAL_CORRECTIONS_CHECK.json) covers the later code-navigation addition. These records retain their original labels and hashes.

V1.0 unifies the manuscript labels and improves repository navigation. Scientific code, saved results, and figures retain their previous bytes. This transition includes no new experiment or PDF compilation.
