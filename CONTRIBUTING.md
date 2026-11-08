# Contributing to PhiMech

Please open a focused pull request against `main` with an explanation of the change, its test evidence and any affected numeric CAD gates. Never commit a production design, private service configuration, licensed vendor SDK, or real site data.

## Development

```sh
uv sync --extra dev
uv run --extra dev pytest -q
uv run --extra dev python -m pyflakes phimech tests
uv build
```

OpenSCAD and FreeCAD are optional, separately installed subprocesses. Tests depending on unavailable kernels should skip explicitly; never report an unrendered result as a passing geometry gate. Preserve the vendored BOSL2 BSD-2-Clause license and the three.js MIT license when changing third-party assets. Code changes should maintain Python 3.9 compatibility in the dependency-free core and keep IR and CAD kernels behind the CSG AST boundary.

## DCO

Sign off every commit with `git commit -s` to certify your contribution under the [Developer Certificate of Origin](https://developercertificate.org/). No CLA is required.
