# Data and code availability

## Data availability

Representative carbon-element characterization maps, retained generated-image examples, structural and physical-response records, and paired section-field inputs are distributed through [the GitHub repository](https://github.com/Yutao-ml/CCC-Generative-design). File definitions, units, source hashes and processing rules are documented in `docs/data_dictionary.md`, `docs/agent_methods.md` and `data/agent/metadata.json`. The response values in the paired agent table are preserved from the retained screening records.

## Code availability

Code for DME-GAN inference and new agent-assisted training runs, frozen random-Fourier-feature ensemble feedback, event-triggered control, controller replay, morphology analysis and retained-response screening is distributed in [the same repository](https://github.com/Yutao-ml/CCC-Generative-design). The selected epoch-270 inference weights, parameters, dependencies and runnable entry points are included. Training outputs are written to new output directories. The released workflows use retained finite-element labels; external solver automation is managed separately.
