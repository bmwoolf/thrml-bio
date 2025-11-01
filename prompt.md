GLOBAL DO-NOTS

- never include personal or sensitive information (keys, tokens, pii). fabricate nothing.
- no emojis.
- do not add a “summary” .md file when making large code changes.
- avoid network calls/telemetry unless explicitly asked.

STYLE & CODE CONVENTIONS
- prefer simple, performant solutions over clever ones; keep hot paths tight.
- enforce DRY: deduplicate logic via helpers/modules; no repetition across files.
- modularity + encapsulation: thin public interfaces, hide internals, minimize side effects
- comments are lowercase; capitalize proper nouns and hardware/software acronyms (CPU, GPU, CUDA, NumPy), no periods, keep them simple and in layman terms as much as possible 
- python identifiers must use snake_case for this repo (override pep8-naming as needed).
- no global mutable state; pass config explicitly.