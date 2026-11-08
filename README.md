# PhiMech

Component-driven enclosure synthesis using OpenSCAD and FreeCAD, with a Python CLI, optional MCP tools, and a local 3D viewer.

![Status: experimental](https://img.shields.io/badge/status-experimental-yellow) ![License: Apache--2.0](https://img.shields.io/badge/license-Apache--2.0-blue)

Created at [Exowatt](https://exowatt.com). The exported core is vendor-neutral; no company service, production design or deployment configuration is needed.

## Status

Experimental: CAD output must be inspected and validated by a qualified engineer before fabrication. The original product-specific examples, site worker, drawing standard and hosted installer are **not** included here.

## Install

Requires Python 3.10+ for development extras (the dependency-free core supports Python 3.9+). In a clone:

```sh
uv sync --extra dev
uv run phimech --version
```

For rendering, install OpenSCAD and/or FreeCAD separately from their official distributions and check availability with `uv run phimech doctor`. OpenSCAD generates SCAD/STL/PNG; STEP/FCStd requires FreeCAD. The `check` command below runs without either kernel.

## Quickstart

```sh
uv run phimech ir schema --json
uv run phimech synth check examples/parametric-box.json
# With OpenSCAD installed:
uv run phimech synth build examples/parametric-box.json -o out/box
uv run phimech view out/box --no-open
```

Use `uv run --extra mcp phimech mcp serve` for optional agent tools; `--extra mcp` syncs the MCP dependencies into the environment first (see `uv run phimech mcp --help` for commands). `uv run --extra dev pytest -q` runs the tests. Separate CAD-kernel tests are skipped when those executables are unavailable.

## Configuration

No remote service is required. The CLI and viewer bind to localhost by default. Override CAD executable paths with `OPENSCAD_BIN` and `FREECADCMD_BIN` as described by `phimech doctor`; `PHIMECH_PRINTER_URL` supplies an optional, user-chosen Moonraker endpoint for explicit printer queries. No printer is contacted by a normal build. An optional external model provider can be integrated outside this core; no gateway or company-specific agent configuration is bundled.

## Third-party components

The vendored BOSL2 OpenSCAD library retains its BSD-2-Clause license at `phimech/vendor/BOSL2/LICENSE`; the viewer vendors three.js under MIT. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [NOTICE](NOTICE). OpenSCAD (GPL) and FreeCAD (LGPL) run as separate, separately installed processes, not as included libraries.

PhiMech does **not** install PyMuPDF: `pyproject.toml` has no `pymupdf`/`fitz` dependency. The private SolidWorks drawing-review integration used an optional import of PyMuPDF (AGPL-3.0 or commercial license); it and the old installer archive containing a wheel are excluded. If reintroduced, obtain a license/compliance review and document dependency terms before distribution. For the optional SolidWorks add-in, build from [PhiMech-SolidWorks](https://github.com/couplet-physical-ai/PhiMech-SolidWorks) with a licensed SolidWorks seat and its locally installed vendor SDK/interop DLLs; no DLLs or installer ZIP are bundled here.

## License

Original source is [Apache-2.0](LICENSE); dependencies and vendored code keep their own licenses as documented above. See [NOTICE](NOTICE).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Sign off commits with DCO (`git commit -s`); no CLA is required. Do not submit production designs, company/site details or licensed SDK binaries.

## Security

Report vulnerabilities via [GitHub private vulnerability reporting](https://github.com/couplet-physical-ai/PhiMech/security/advisories/new), not a public issue.
