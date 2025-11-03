GLOBAL DO-NOTS

- never include personal or sensitive information (keys, tokens, pii). fabricate nothing
- no emojis
- do not add any summary or report .md files summarizing your changes when making large code changes
- do not open multiple terminal windows to run in either background or foreground. keep everything in 1 terminal max unless i specify otherwise. reuse that terminal window

STYLE & CODE CONVENTIONS
- prefer simple, performant solutions over clever ones; keep hot paths tight
- enforce DRY: deduplicate logic via helpers/modules; no repetition across files
- modularity + encapsulation: thin public interfaces, hide internals, minimize side effects
- comments are lowercase; capitalize proper nouns and hardware/software acronyms (CPU, GPU, CUDA, NumPy), no periods, keep them simple and in layman terms as much as possible 
- python identifiers must use snake_case for this repo (override pep8-naming as needed)
- when pushing code, commit each file individually with descriptions of changes in <=7 words, all lowercase, except for names/products
- clean up and remove all temporary scripts/.md files
- if making code changes, test before committing and pushing
- never suggest skipping something if i deem it necessary to investigate. solve root cause problems by asking 'why' until arriving that the specific cause that is resulting in the undesirable effect
- leave imports at the top of each file, rather than importing at the function level