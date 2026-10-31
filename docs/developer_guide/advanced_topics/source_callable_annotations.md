# Annotations during private callable preparation

The private selected-dtype preparation supports ordinary nongeneric CPython 3.14
functions with deferred annotations. Reading a callable preimage, preparing its
body, and checking a preparation record do not evaluate its annotation callback.
The original code, defaults and annotation storage are retained. Earlier Python
versions retain their existing eager annotation handling.

The compatibility adapter matches CPython 3.14's complete ordered
`func_traverse` references before observing annotation storage. The measured
modern forms are an absent cache and a cached dictionary, including dictionary
subclasses and aliases to other function metadata. Duplicate references are
significant. Unknown layouts, raw tuple caches, generic headers and non-function
annotation callback owners require separate support; the adapter does not fall
back to an evaluating annotation getter or a memory-layout offset.

Native dictionary snapshots avoid annotation mapping overrides during cache
observation and record checks. Alias stress controls qualify this adapter
boundary; they do not establish that every unrelated runtime-owner mapping hook
is inert throughout the larger preparation process.

A prepared function keeps its annotations deferred until an explicit metadata
read. Its first VALUE read obtains the original function's dictionary and records
its identity and native key/value bindings before returning it. Prepared versions
of the same original share this controlled transition. Later replacement,
reordering or binding changes invalidate their records. Referenced mutable
objects are not recursively frozen. The registry holds weak keys and weak values.

An original-side first annotation read invalidates an existing cold record and
requires fresh preparation. A passive check cannot safely distinguish an initial
evaluation from an evaluation followed by an unobserved dictionary mutation, so
it never adopts that transition. Annotation exceptions propagate unchanged while
the protected state stays cold; metadata changes during a callback leave the old
record stale. Reentrant materialization is unsupported.

The prepared callback delegates public FORWARDREF and STRING formats through
`annotationlib.call_annotate_function` using the original callback and owner.
This preserves the compiler callback's namespace and fake-globals fallback.
The internal VALUE_WITH_FAKE_GLOBALS format is not implemented by the forwarding
closure. Native qualification must exercise `annotationlib.get_annotations`,
including unresolved names and raising expressions, in addition to direct VALUE
reads. These checks establish private preparation behavior, not numerical export
acceptance or terminal CI success.

The native contracts are documented in CPython's
[function implementation](https://github.com/python/cpython/blob/v3.14.8/Objects/funcobject.c)
and [annotationlib implementation](https://github.com/python/cpython/blob/v3.14.8/Lib/annotationlib.py).
