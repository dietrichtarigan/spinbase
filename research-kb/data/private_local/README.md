# Private local data (air-gapped)

Put unpublished MOKE images/notes, MuMax3 `.mx3` scripts and `table.txt` logs, and lab notes here,
then run `kb private data/private_local/<folder> --material Fe3GaTe2`.

Everything ingested from this folder is tagged `is_private=True`, fingerprinted, and hard-blocked
from any outbound request (DeepSeek, arXiv, OpenAlex, Semantic Scholar). This folder is git-ignored.
