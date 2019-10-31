# Contribution guidelines:

## Adding a function/class/instance:
- Docstrings
- Name in `__all__`
- Links in `README.md` and `index.rst`
- Update `CHANGELOG.md`
- In case of new dependencies add them to `README.md`, `index.rst` and
  `setup.py`
- Write tests (if applicable)
- Rebuild the docs

## Adding a new module:
- Include encoding, docstring and `__all__`
- Create doc file `<modulename>.rst` in `docs/source`
- Links in `README.md` and `index.rst` (x2 - Contents and Modules)
- Update `CHANGELOG.md`
- Import in `__init__.py`
- In case of new dependencies add them to `README.md`, `index.rst` and `setup.py`
- Write tests (if applicable)
- Rebuild the docs

## New version:
- Modify version number in `__info__.py`
- Update `CHANGELOG.md`
- Create and push git tag
- Rebuild the docs
