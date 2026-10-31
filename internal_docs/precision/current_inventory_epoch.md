# Current inventory after main #279

The current live metadata inventory is
[`source_migration_inventory_current.json`](source_migration_inventory_current.json):
**1,856 raw rows / 3,712 caller obligations**. The historical
[`source_migration_inventory.json`](source_migration_inventory.json) remains
byte-identical at SHA-256
`b875cf3a8169c2919c425eb9d5f11fda9396020945d7ffc4164fb748e153474c`, with its
**1,852 / 3,704** epoch and historical replay adapters preserved. Neither manifest
is a SOURCE or numerical acceptance result.

Main #279 adds four rows: same-dtype Bitcast, dominant/fully-masked attention,
FP16 masked attention, and weak-integer Where. Five existing attention ordinals
and two Where ordinals move. Two Bitcast rows change from the original opset 23
request to explicit 26 with numerical checks; seven attention graph callbacks
change. The original opset 23 contracts remain recorded and receive no acceptance
from an opset 26 result. The incoming BF16 capability matrix also adds two
variants, separately from raw rows; masked BF16 attention has no numerical claim.

The complete predecessor-to-current
[`source_inventory_epoch_mapping.json`](source_inventory_epoch_mapping.json)
records every original origin/ordinal, complete contract hash, current target
identity, changed fields and reviewed relocation. It retains all original manual
identities, **752 anchors**, **104 metadata-free registrations**, all **50 optional
models (45 MaxText / five MaxDiffusion)** and all **63 historical opaque recipe
obligations**. Seven Equinox recipes now have explicit factory metadata, leaving
56 currently opaque recipes; that representation change does not accept their
SOURCE implementation or numerics. Three added registrations have zero raw
testcases (`call_hi_primitive`, `platform_index`, `polynomial`) and remain explicit
registry obligations rather than invented rows.

Reconciliation covers the entire historical interval, including earlier
elementary-function metadata and Equinox factory changes. There are 27 changed
core canonical contracts and 45 MaxText raw config-path relocations. MaxText's
registration source also changes its initializer recipe; its unchanged 45 labels
do not transport prior construction/execution proof. Exact source, config,
observation, receipt and environment identities remain separate in the mapping.

ROOT's guarded complete observations used the genuine clean upstream pins
MaxText `17d805e3488104b5de96bd19be09491ff73c57c1` and MaxDiffusion
`b4f95730bf4f00c4fd9e3dd2fdda1b50484afda2`. The modern observation has 1,856 rows;
the older observation has 1,855 because its JAX lacks the retained conditional
`one_minus_square` row. All shared canonical contracts agree except
`examples.lax/remat2`: modern JAX's captured checkpoint closure adds empty
`static_argnames`, whereas older JAX lacks that field. Both exact full contracts
and their receipts are recorded. The strict older whole-inventory comparison
continues to report that canonical mismatch as pending audit; no field is
normalized, ignored or granted semantic equivalence.

The current manifest composes the genuine modern metadata observation with a
later manual-only AST capture after this repair's test additions. Final guarded
recollection independently confirms the complete parsed modern manifest,
including **316 manual files / 2,693 functions**; serialization bytes differ.
Older recollection confirms the same manual identities, all 50 optional rows,
the one retained conditional absence and exactly the documented remat2 mismatch.
No source, runtime or numerical evidence transfers through this mapping.

`scripts/source_precision_inventory.py` defaults to the current manifest.
`scripts/source_precision_probe.py` explicitly keeps the historical manifest and
historical adapter pair; historical replay/corpus assertions and strict
`_resolve_case` drift guards remain. The complete live collection test uses the
current manifest and retains full raw-object/contract preservation and the
original 18 selected preflights (36 cells). A current replay lane with matching
adapters and broader cohort proof remains open B–F work.

Published merge head `a69d0730`, CI run 541, failed the actual Python 3.14 live
inventory membership assertion. This repair addresses that observed mismatch;
the hosted cast case had passed and is a separate result. All seven pinned
all-files hooks pass. Eleven complete affected owners pass **1,111 cases on
Python 3.14/JAX 0.12 and 1,114 on Python 3.11/JAX 0.10**, without failures, errors or
skips. Their 1,102 common IDs retain identical order; the existing native-wrapper
owner supplies the 9 modern-only/12 older-only runtime-dependent parameters.
All 49 original inventory-owner IDs remain. Guarded source/environment fences
and process cleanup hold. Independent review accepts this finite repair;
all eight checks succeeding on its exact published head remain required.
Public LEGACY defaults, original inputs/criteria, broader
symbols/options/opsets, original22/B=5 resource work, complete model/numerical
qualification and the coordinated public transition remain open. This inventory
repair neither completes nor reduces B–F scope.
