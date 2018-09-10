# data_tools version history

## v0.0.7:
- `test_data_tools` has tests for almost all classes and functions.
  Functions that include plots are skipped (cannot be tested). Only test
  missing is `models.Lasso`.
- In `models` module:
    - Class `DoseResponse` improved and bugs fixed.
- In `databases` module:
    - Function `op_kinase_substrate` now can also download
      dephosphorylation interactions.
- In `plots` module:
    - Function `venn` now can plot global percentages instead of
      absolute counts.
## v0.0.6:
- Renamed `sets` to `iterables`.
- In `models` module:
    - Class `DoseResponse`: Wrapper class for
      ``scipy.optimize.least_squares`` to fit dose-response curves on a
      pre-defined Hill function.
- In `sets` module:
    - Function `unzip_dicts`: Unzips the keys and values for any number
      of dictionaries passed as arguments (see below for examples).
    - Function `chunk_this`: For a given list *L*, returns another list
      of *n*-sized chunks from it (in the same order).
- In `plots` module:
    - Function `venn` now accepts up to 5 sets.
- In `databases` module:
    - Function `op_kinase_substrate`: Queries OmniPath to retrieve the
      kinase-substrate interactions for a given organism.
    - Function `kegg_link`: Queries a request to the KEGG database to
      find related entries using cross-references.
    - Function `kegg_pathway_mapping`: Makes a request to KEGG pathway
      mapping tool according to a given pathway ID. The user must
      provide a query of IDs to be mapped with their corresponding
      background colors (and optionally also foreground colors).
- Added `diffusion` module:
    - Function `euler_explicit1D`: Computes diffusion on a 1D space over
      a time-step using Euler explicit method.
- Renamed `databases.up_query` to `databases.up_map`

## v0.0.5:
- Added `databases` module:
    - Function `up_query`: Queries a request to UniProt.org in order to
      map a given list of identifiers.
- Linked GH-pages to HTML documentation.
- Added changelog.
- Implemented unit test.

## v0.0.4
- In `plots` module:
    - Function `density`: Generates a density plot of the values on a
      data frame (row-wise).
    - Function `venn`: Plots a Venn diagram from a list of sets *N*.
      Number of sets must be between 2 and 4 (inclusive).
- In `sets` module:
    - Function `subsets`: Function that computes all possible logical
      relations between all sets on a list *N* and returns all subsets.
      This is, the subsets that would represent each intersecting area
      on a Venn diagram.
    - Function `multi_union`: DEPRECATED.

## v0.0.3
- Added `models` module:
    - Class `Lasso`: Wrapper class inheriting from
      ``sklearn.linear_model.LogisticRegressionCV`` with L1
      regularization.
        - Function `fit_data`: Fits the data to the logistic model.
        - Function `plot_score`: Plots the mean score across all folds
          obtained during CV. The optimum C parameter chosen and its
          score are highlighted.
        - Function `plot_coef`: Plots the non-zero coefficients for the
          fitted predictor features.
- In `plots` module:
    - Function `piano_consensus`: Generates a GSEA consensus score
      plot like R package ``piano``'s ``consensusScores`` function, but
      prettier. The main input is assumed to be a ``pandas.DataFrame``
      whose data is the same as the ``rankMat`` from the result of
      ``consensusScores``.

## v0.0.2
- Documentation also available in PDF.
- Added `plots` module:
    - Function `volcano`: Generates a volcano plot from the differential
      expression data provided.

## v0.0.1
- Automatic documentation implemented (HTML generated).
- Package can now be installed.
- Added `sets` module:
    - Function `bit_or`: Returns the bit operation OR between two
      bit-strings a and b. NOTE: a and b must have the same size.
    - Function `find_min`: Finds and returns the subset of vectors whose
      sum is minimum from a given set A.
    - Function `in_all`: Checks if a vector x is present in all sets
      contained in a list N.
    - Function `multi_union`:  Returns the union set of all sets
      contained in a list N.
- Added `strings` module:
    - Function `is_numeric`: Determines if a string can be considered a
      numeric value. NaN is also considered, since it is float type.
    - Function `join_str_lists`: Joins element-wise two lists (or any 1D
      iterable) of strings with a given separator (if provided). Length
      of the input lists must be equal.
