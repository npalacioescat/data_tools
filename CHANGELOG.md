# data_tools version history

## v0.0.9 (WIP):
- In `databases` module:
    - Added option to retrieve gene symbols instead of UniProt AC in
      `op_kinase_substrate` function.
- In `diffusion` module:
    - Function `coef_mat_hetero`: Builds a block tri-diagonal
      coefficient matrix for a n-dimensional diffusion problem with
      heterogeneous diffusion coefficients.
- In `models` module:
    - Class `Linear`: Linear regression model using least squares.
    - Class `PowerLaw`: Fits a power law model to the provided data.
- In `plots` module:
    - Added option to show labels on significant points of `volcano`.
    - Function `phase_portrait`: Generates a phase portrait of a ODE
      system given the nullclines.
    - Function `chordplot`: Generates a chord plot from a collection of
      nodes and edges (and their sizes).
    - Function `PCA`: Computes the principal component analysis (PCA)
      and plots the results.

## v0.0.8:
- Added `signal` module:
    - Function `fconvolve`: Convolves two vectors or arrays using Fast
      Fourier Transform (FFT).
    - Function  `gauss_kernel`: Returns a N-dimensional Gaussian kernel.
    - Function `gauss_noise`: Applies additive Gaussian (white) noise to
      a given signal.
- In `spatial` module:
    - Function `equidist_polar`: For a given number of points (and
      optionally radius), returns the Cartesian coordinates of such
      number of equidistant points (in polar coordinates).
- In `plots` module:
    - Function `cluster_hmap`: Generates a heatmap with hierarchical
      clustering dendrograms attached.
    - Function `upset_wrap`: Wrapper for UpSetPlot package. Mostly just
      generates the Boolean multi-indexed ``pandas.Series`` the
      ``upsetplot.plot`` function needs as input.
    - Function `similarity_heatmap`: Given a group of sets, generates a
      heatmap with the similarity indices across each possible pair.
    - Function `similarity_histogram`: Given a group of sets, generates
      a histogram of the similarity indices across each possible pair
      (same-element pairs excluded).

## v0.0.7:
- Support for Python 3.x versions.
- `test_data_tools` has tests for almost all classes and functions.
  Functions that include plots are skipped (cannot be tested). Tests
  missing are `models.Lasso` and `spatial` module (or maybe more).
    - Test suite has been improved and continuous integration with
      TravisCI has been implemented.
- Added `spatial` module:
    - Function `neighbour_count`: Given an array (up to three
      dimensions), returns another array with the same shape containing
      the counts of cells' neighbours whose value is zero.
    - Function `get_boundaries`: Given an array, returns either the mask
      where the boundary edges are or their counts if specified.
- In `iterables` module:
    - Function `similarity`: Computes the similarity index between two
      sets.
- In `diffusion` module:
    - Function `build_mat` unifies coefficient matrix building for any
      number of dimensions, numerical method and boundary condition.
    - ~~Function `euler_explicit2D`: Computes diffusion on a 2D space
      over a time-step using Euler explicit method.
    - Function `euler_implicit_coef_mat`: Computes the coefficient
      matrix to solve the diffusion problem with the Euler implicit
      method.
    - Function `crank_nicolson_coef_mats`: Computes the coefficient
      matrices to solve the diffusion problem with the Crank-Nicolson
      method.
    - Function `build_coef_mat`: Builds a coefficient matrix according
      to the central and neighbor coefficients, system size and boundary
      conditions.~~
- In `models` module:
    - Class `DoseResponse` improved and bugs fixed.
- In `databases` module:
    - Function `op_kinase_substrate` now can also download
      dephosphorylation interactions.
- In `plots` module:
    - Function `venn` now can plot global percentages instead of
      absolute counts. Also added the option to show the set sizes on
      the legend.
    - Function `density` now can handle data frames with samples on the
      columns.
    - Created 3 custom colormaps: `cmap_bkgr` (black-green), `cmap_bkrd`
      (black-red) and `cmap_rdbkgr` (red-black-green).

## v0.0.6:
- Renamed `sets` module to `iterables`.
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
- Python 3 compatibility

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
