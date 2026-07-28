# WheatMMDL

Multimodal deep learning with pooled across-season and within-season cross-validation,
integrating genomic, NIR, drone, weather, and secondary-trait data to predict grain yield
and quality in bread wheat.

## Overview

This repository contains deep-learning and linear-model workflows used to evaluate
trait-specific combinations of data sources in bread wheat. The models combine one or more
of the following modalities:

- Genomic markers
- Near-infrared spectroscopy (NIR)
- Drone-derived features
- Weather variables
- Secondary phenotypic traits

The deep-learning workflows use a multimodal multilayer perceptron (`MMMLP`).
Hyperparameters are selected with Bayesian optimization. The repository also includes
linear benchmark models fitted with the R package `BGLR`.

## Cross-validation schemes

Two evaluation schemes are included:

1. **Pooled across-season CV:** observations from all available seasons are analyzed together.
2. **Within-season CV:** models are evaluated separately within each year.

The deep-learning schemes are available with or without secondary traits as an additional
input modality. Two BGLR scripts provide the corresponding linear-model analyses.

## Scripts

| Script | Model | Cross-validation | Secondary traits |
|---|---|---|---|
| `Models-v8-v6-CV1-10CV-Basic_code` | MMMLP | Pooled across seasons | No |
| `Models-v8_v6- traits_covs-Basic_code` | MMMLP | Pooled across seasons | Yes |
| `Models-v8-v6-by_year-Basic_code` | MMMLP | Within season | No |
| `Models-v8-v6-by_year-traits_covs-Basic_code` | MMMLP | Within season | Yes |
| `Model_BGLR- Basic code.r` | BGLR linear model | Reference analysis | No |
| `Model_BGLR-Basic code- trait covs.R` | BGLR linear model | Reference analysis | Yes |

## Data

The scripts read an `.RData` file containing the following objects:

| Object | Description |
|---|---|
| `pheno` | Phenotypes, years, plots, and genotype identifiers |
| `markers` | Genomic marker matrix |
| `nirs` | NIR-derived predictors |
| `df_smm_drones` | Drone-derived predictors |
| `weather_smms` | Weather predictors used by the pooled multimodal workflow |

Rows in `pheno`, `nirs`, and `df_smm_drones` are matched using:

```text
year_genotype
```

The fold-assignment CSV files must contain the columns used by the selected script, such as
`year`, `trait`, `trait_no`, and `fold`.

Data are not included in this repository. Update `dir_datasets` so that it points to the
directory containing the required `.RData` files.

## Requirements

- CPython 3.11 on Windows
- NumPy
- pandas
- PyTorch
- scikit-learn
- pyreadr
- bayesian-optimization

The linear-model scripts additionally require:

- R
- BGLR

Install the public dependencies with:

```bash
pip install numpy pandas torch scikit-learn pyreadr bayesian-optimization
```

Install BGLR from R with:

```r
install.packages("BGLR")
```

The project-specific Python modules are distributed as compiled `.pyc` files in:

```text
Pycs_Win_3.11/
```

The directory contains the compiled versions of the required utility and MMMLP modules.
Because `.pyc` files are Python-version specific, use CPython 3.11 on Windows and preserve
the provided directory structure.

The scripts load these modules using paths relative to the repository:

```python
dir = os.getcwd()
dir_utils_AML = "Pycs_Win_3.11/"
dir_progs = "Pycs_Win_3.11/"
sys.path.append(os.path.normpath(dir_utils_AML))
sys.path.append(os.path.normpath(dir_progs))
```

Run the scripts from the repository root so that `Pycs_Win_3.11/` can be found.

## Configuration

Each script contains a short `# Example` section. Edit these values before running a model.

### Pooled model without secondary traits

```python
# Example
trait = "gpc"
predictor = "genotype"
```

### Pooled model with secondary traits

```python
# Example
trait = "bbch59_gdd"
traits_covs = ["bbch10_gdd", "bbch30_gdd"]
predictor = "genotype"
```

### Within-season model

```python
# Example
datanumber = 0
trait_no = 0
trait = "gpc"
predictor = "genotype"
```

Predictor combinations can be written with `+`, for example:

```python
predictor = "genotype+NIRs+Drone"
```

In scripts using secondary traits, `Traits_covs` is automatically added to the selected
predictor combination.

## Running the code

Run a deep-learning script from the command line:

```bash
python Models-v8-v6-CV1-10CV-Basic_code.py
```

For example, to run the within-season model with secondary traits:

```bash
python Models-v8-v6-by_year-traits_covs-Basic_code.py
```

The code automatically uses a CUDA-compatible GPU when one is available; otherwise, it runs
on the CPU.

Run a BGLR model with:

```bash
Rscript "Model_BGLR- Basic code.r"
```

## Outputs

The scripts create CSV files containing:

- Observed and predicted values for each fold
- Prediction metrics by environment
- Overall correlation, MSE, and normalized RMSE
- Matching among the top-performing genotypes in the within-season analyses
- Selected hyperparameters
- Execution time and model settings

Outputs are saved under directories organized by dataset, model, trait, and, when applicable,
year.

## Reproducibility

The basic configuration uses:

```text
10 outer folds
10 inner folds
ReLU activation
Residual connections
Bayesian hyperparameter optimization
Seed = 42
```

Fold-assignment files should be retained and reused to reproduce the reported partitions.

## Citation

If you use this code, please cite the associated study:

> *Multimodal deep learning identifies trait-specific data sources for predicting yield and
> grain quality in bread wheat across seasons.*

Complete citation details can be added here after publication.
