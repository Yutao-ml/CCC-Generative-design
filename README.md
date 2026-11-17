# Carbon-ceramic DME-GAN examples

Example resources for **Adaptive electrothermally guided generative design of carbon microstructures in ceramic composites**.

This repository provides inference with a selected dual-scale morphology-envelope-constrained generative adversarial network (DME-GAN), morphology-descriptor calculations and candidate-response screening. Version 1.1.0 contains saved image examples, retained response data and the corresponding analysis code.

## Included data

- 32 carbon-element characterization maps.
- 20 saved generated fields each for autoregressive (AR), variational autoencoder (VAE), diffusion and generative adversarial network (GAN) models.
- 500 saved DME-GAN fields from the selected epoch-180 screening library.
- 302 physical-response records: 300 candidate entries and two benchmark records.
- A 300-record table containing 21 structural features and the retained physical responses.
- Four-model settings, material-input tables and response-model settings.

The image subsets use the first saved entries in numeric order. Their original identifiers and SHA-256 hashes are in `data/image_index.csv`. The three 100-entry response cohorts may overlap. The saved examples are not used to re-estimate the article's population-level improvements.

![Saved field examples](docs/preview.png)

## Run the examples

Use Python 3.12. Open a terminal in the repository root and run:

```text
python -m pip install -r requirements.txt
python scripts/verify_manifest.py .
python scripts/run_examples.py --out results/example_run --device cpu
```

The default run calculates descriptors for all 612 saved images, screens the 300 candidates against the two benchmarks, and generates 20 new DME-GAN fields. To generate 50 new fields instead:

```text
python scripts/run_examples.py --out results/example_50 --device cpu --count 50
```

The sampling example supports 1-100 fields per run. New fields, float32 arrays and random-state information are saved separately from the retained data. Use a new or empty output directory.

For saved-data analysis without installing PyTorch:

```text
python -m pip install -r requirements-analysis.txt
python scripts/run_examples.py --out results/analysis --skip-generation
```

## Inputs and outputs

- `data/characterization/`: reference maps for image preparation and generator-reference initialization.
- `data/generated_examples/{ar,vae,diffusion,gan,dmegan}/`: saved generated-image subsets.
- `data/checkpoint/selected_generator.pt`: selected generator weights, with state-tensor values unchanged.
- `data/responses/`: transient, mechanical and structure-response tables.
- `data/model_settings/` and `configs/`: reported settings and inference configuration.
- `data/derived/`: precomputed descriptor summaries and response-screening results for the released data.
- `src/dmegan_examples/`: generator architecture, image descriptors and screening functions.
- `docs/data_dictionary.md`: definitions, units and data relationships.
- `validation/`: actual verification records; `MANIFEST.json`: file sizes and hashes.

The run produces `image_descriptors.csv`, `descriptor_group_means.csv`, `response_screening.csv` and `example_summary.json`. New generation also creates a `new_fields/` directory containing PNGs, float32 fields and sampling information.

## Method and scope

The inference loader uses the matching epoch-180 architecture and initializes its reference bank from the 32 characterization maps in numeric order. The checkpoint contains the generator state; optimizer and discriminator states are not required for inference.

Four-model images are saved examples; this release does not implement inference or retraining for those four models. Image statistics describe the included sets. Response screening uses retained numerical outputs, converts milliseconds to seconds for temperature integration, and evaluates the stated electrothermal and modulus benchmarks. The 21-feature table is supplied for inspection and is distinct from the 14 per-image evaluation descriptors.

Full training, checkpoint selection, surrogate-feedback optimization and finite-element re-solving are outside this example release. Installation and running this repository do not reproduce every result in the article.

## Licence and citation

Software is licensed under [MIT](LICENSE). Data and checkpoint weights in `data/` are licensed under [CC BY 4.0](LICENSES/CC-BY-4.0.txt); see [data licensing](data/LICENSE.md). Citation metadata are in [CITATION.cff](CITATION.cff).

Both statements in [AVAILABILITY.md](AVAILABILITY.md) reserve the same [SHARED_ARCHIVE_DOI]. Insert the real version-specific DOI after archival deposit. No live repository address or DOI has been assigned in these files.
