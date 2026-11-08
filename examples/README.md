# Standalone examples

These synthetic, generic intent examples use no production boards or site infrastructure. Check one without CAD binaries:

```sh
uv run phimech synth check examples/parametric-box.json
```

With OpenSCAD installed, build an STL:

```sh
uv run phimech synth build examples/parametric-box.json -o out/box
```

For STEP/FCStd install FreeCAD separately.
