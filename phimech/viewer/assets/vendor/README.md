# Vendored third-party browser modules

These files are committed on purpose: the viewer must work with **no network access**. Nothing
here is fetched from a CDN at runtime.

| File | Project | Version | License |
| --- | --- | --- | --- |
| `three.module.js` | [three.js](https://github.com/mrdoob/three.js) | r169 | MIT |
| `OrbitControls.js` | three.js `examples/jsm/controls/OrbitControls.js` | r169 | MIT |
| `STLLoader.js` | three.js `examples/jsm/loaders/STLLoader.js` | r169 | MIT |

Copyright © 2010-2024 three.js authors. MIT License —
Full license text: [LICENSE.threejs](LICENSE.threejs), copied from upstream r169.

## Do not remove the import map

`OrbitControls.js` and `STLLoader.js` both start with a **bare** specifier:

```js
import { Controls, MOUSE, Quaternion, ... } from 'three';
```

A browser cannot resolve `'three'` on its own. Without the import map in `index.html` the two
modules fail to load and the canvas stays black with only a console error, so `index.html` must
always carry:

```html
<script type="importmap">{"imports":{"three":"./vendor/three.module.js"}}</script>
```

Keep the map before any `<script type="module">` tag — import maps must be parsed first.

## Upgrading

Replace all three files from the same three.js release (mixing revisions breaks the internal
`Controls` base class that `OrbitControls` extends), update the version column above, and re-run
`python3 tests/test_viewer.py`, which asserts the import map and the absence of CDN hosts.
