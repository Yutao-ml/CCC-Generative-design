# Example data and computation definitions

## Images and checkpoint

Characterization PNG files are RGB carbon-element maps. The reference loader extracts the red channel, resizes it bilinearly to 512x512 pixels and divides intensities by 255. The resulting normalized two-dimensional fields are used for morphology analysis. Their numeric filename order is preserved.

The 580 retained generated PNG files are grayscale fields with intensities in [0,255]. New samples are saved both as normalized float32 arrays and as 8-bit PNGs. Descriptor values computed after 8-bit saving can differ from values computed on floating-point fields.

The checkpoint contains the generator state only. `configs/inference.json` specifies the matching inference architecture and latent scale. The released inference checkpoint is selected at epoch 270 and requires no reference-image bank. The retained image data keep their original source labels. The included inference implementation does not perform training or select checkpoints.

## Image descriptors

Descriptors are computed on each normalized field. The binary mask uses the retained 256-bin Otsu calculation and `intensity > max(threshold,0.03)`. Connected components use four-neighbour connectivity. Distances and areas refer to the image grid.

| Descriptor | Definition | Unit |
|---|---|---|
| `mean_intensity` | Pixel mean | 1 |
| `std_intensity` | Population standard deviation | 1 |
| `p95_intensity` | 95th intensity percentile | 1 |
| `area_fraction` | Binary-mask occupancy | 1 |
| `interface_density` | Half the sum of horizontal and vertical neighbour-change fractions | 1 |
| `component_count` | Number of carbon-mask connected components | count |
| `mean_component_area` | Mean component area | pixel^2 |
| `correlation_length` | First horizontal binary-mask lag below exp(-1), capped at 128 pixels | pixel |
| `hist_entropy` | Entropy of the 64-bin intensity histogram using natural logarithms | nat |
| `gradient_mean` | Mean magnitude of the discrete intensity gradient | pixel^-1 |
| `edge_density` | Fraction with discrete gradient magnitude greater than 0.05 | 1 |
| `autocorr_x` | Normalized one-pixel horizontal intensity autocorrelation | 1 |
| `autocorr_y` | Normalized one-pixel vertical intensity autocorrelation | 1 |
| `component_area_max` | Largest component area | pixel^2 |

The example script reports per-image descriptors and arithmetic group means for six groups: characterization, AR, VAE, diffusion, GAN and DME-GAN. Their saved-set sizes and source populations differ. These descriptive summaries do not re-estimate the article's full model-comparison statistics.

## Response data

`thermal_history.csv` contains 0-50 ms records at 10 ms intervals. `Tmax_C` and `Tmin_C` are extrema in degrees Celsius; `DeltaT_K` is their difference in kelvin. `Jmean_A_cm2` and `Jmax_A_cm2` are current-density responses in A/cm^2. `mechanical_response.csv` contains the compression displacement in micrometres and effective compressive modulus in GPa.

The response data contain all 300 candidate records and two benchmark records. The candidates comprise three retained sampling cohorts of 100 entries each. Cohort labels identify selection methods, not independent physical experiments. Only the response columns consumed by these examples are included. The included tables are retained numerical results; running the examples does not solve a finite-element model.

The response quantities used for screening are the mean and maximum current density at 20 ms, the time integral of `DeltaT_K`, and the time integral of `Tmax_C - 20`. Time is converted from milliseconds to seconds before trapezoidal integration, giving K*s for both integrals. The compressive modulus can be checked from a 1 MPa load over a 60 micrometre loading length: `E_GPa = 0.001*60/abs(displacement_um)`.

For each record, the electrothermal ratio is the maximum of the four response ratios relative to the electrothermal reference. The modulus ratio is the modulus divided by that of the modulus reference. Electrothermal and modulus conditions are evaluated as ratios <= 1 and >= 1, respectively. Reference records are reported separately from candidates. These classifications describe this fixed example subset and its stated benchmarks.


## Saved image subsets and source index

The AR, VAE, diffusion and GAN folders each contain the first 20 saved evaluation images in numeric order. DME-GAN contains the first 500 entries from the retained screening library associated with the selected epoch-180 generator. The archive distinguishes these populations and does not substitute one for the article's morphology-comparison evaluation population. `data/image_index.csv` records each image's group, original identifier and SHA-256.

## Structure-response table

`structure_response.csv` contains 300 retained records. Seven section-level quantities are summarized by the across-section mean, population standard deviation and range, forming 21 features. The quantities are normalized intensity mean and standard deviation, binary carbon-mask occupancy, uniformity, aggregation, interface density and normalized radial correlation length. The suffix `section_sd` means population standard deviation across representative sections.

For this retained table, uniformity is 1/(1+CV) on a 10x10 partition; aggregation is the largest eight-connected carbon component divided by carbon area; interface density uses a 3x3 morphological gradient; radial correlation length is normalized by half the image width. These seven screening descriptors are separate from the 14 per-image evaluation descriptors described above. The table is provided for inspection; this release does not fit a new surrogate or perform a generator-feedback experiment.

The response columns give average and maximum current density at 20 ms (A/cm^2), integrated temperature range and maximum temperature rise (K*s), and effective compressive modulus (GPa). `Keff_GPa` is the retained column identifier for compressive modulus. `run_id` links each row to the transient and mechanical response tables. Cohorts can share assemblies; row count is not an independent-experiment count.

## Settings reported in the manuscript

`data/model_settings/four_model_settings.csv` and `material_parameters.csv` reproduce the corresponding settings and material-input tables in the audited manuscript. `configs/response_parameters.json` records the retained domain, excitation, heat-transfer, output-interval and compression settings. The constitutive inputs are prescribed equivalent-property values. The example script reads retained outputs and does not rerun these finite-element models.


## Precomputed examples

`data/derived/image_descriptors.csv` contains 14 descriptors for each of the 612 saved fields. `descriptor_group_means.csv` reports their within-set arithmetic means and image counts. `response_screening.csv` records the recomputed 302 response endpoints and benchmark classifications. These files are generated by the released example script from the included data; the group means describe these saved sets rather than the article's full comparison populations.

## Checkpoint storage

The generator-state export is stored as three ordered binary parts of 20,000,000, 20,000,000 and 8,204,771 bytes. `selected_generator.pt.parts.json` records each part's size and SHA-256 and the combined size and SHA-256. The loader checks and reassembles the parts in memory. All 64 state entries match the source epoch-270 generator. The inference export omits discriminator and optimizer states. An optional `python scripts/restore_checkpoint.py --out results/selected_generator.pt` command restores the export to a standalone file.


## Agent feedback inputs

`data/agent/` supplies 300 paired sets of four retained interpolation sections. Six NPZ files store normalized float64 intensities at 100x100 points per section and the corresponding run identifiers. Arrays use `record, section, y, x` order. Numeric values are parsed from the retained solver-input text fields without spatial filtering. Source text-file SHA-256 hashes, section order and NPZ positions are listed in `section_sources.csv`.

`feedback_soft21.csv` contains smooth descriptor inputs calculated by the released feedback encoder, with unchanged maximum-current-density and thermal-integral targets linked by `run_id`. The `feature_schema` field identifies this encoding. The existing 21-column screening table retains its original descriptor definitions. Paired records can reuse sections and retain the original sampling cohorts.

The response scales 135 A/cm^2 and 0.85 K*s normalize the two channels before the ensemble-dispersion contribution and softplus penalty are calculated. Further algorithm definitions and run instructions are in `agent_methods.md`.
