# Third-party notices

PhiMech is licensed under Apache-2.0 (see [`LICENSE`](LICENSE)). It redistributes the following
third-party software under the licences stated here. Each entry says where the copy lives, which
upstream revision it is, and how it is used, so the attribution can be checked rather than trusted.

## BOSL2 — The Belfry OpenSCAD Library, v2

* **Upstream:** <https://github.com/BelfrySCAD/BOSL2>
* **Copy:** [`phimech/vendor/BOSL2/`](phimech/vendor/BOSL2/) — the `*.scad` library files and
  `LICENSE`, unmodified. The pinned commit and refresh procedure are in
  [`phimech/vendor/BOSL2/VENDORED.md`](phimech/vendor/BOSL2/VENDORED.md).
* **Use:** included at render time by the OpenSCAD backend (`include <BOSL2/…>` through
  `OPENSCADPATH`) when a design uses the `scad_module` CSG node. It is never linked into, or
  translated into, PhiMech's Python.
* **Licence:** BSD 2-Clause. Full text follows.

```
BSD 2-Clause License

Copyright (c) 2017-2019, Revar Desmera
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## three.js

The 3D viewer vendors three.js r169 (MIT): `three.module.js`, `OrbitControls.js`,
and `STLLoader.js`. Copyright © 2010–2024 three.js authors. The complete MIT
license text, copied from the upstream r169 release, ships beside the modules at
[`phimech/viewer/assets/vendor/LICENSE.threejs`](phimech/viewer/assets/vendor/LICENSE.threejs).
The version and module inventory are documented in that directory's README.

OpenSCAD and FreeCAD are separately installed subprocesses, not redistributed
software in this repository. A vendor STEP tray of unknown provenance and the
old SolidWorks installer archive with its nested AGPL/commercial PyMuPDF wheel
are not included in the export.
