# data_tools version history

## v0.0.5 (WIP):
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