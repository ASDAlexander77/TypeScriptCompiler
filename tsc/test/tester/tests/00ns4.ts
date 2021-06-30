let glb1 = 0;

namespace fooX.bar.baz {
    let glb1 = 1;

    export function b() {
        glb1++;
        assert(glb1 == 2, "fooX.bar.baz.glb1");
    }
}

import bz = fooX.bar.baz;
function main() {
    bz.b();
    assert(glb1 == 0, "glb1");
    print("done.");
}
