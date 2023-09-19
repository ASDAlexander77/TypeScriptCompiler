let glb1 = 10;
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
    assert(glb1 == 10, "glb1");
    assert(bz.glb1 == 2, "glb1");
    print("done.");
}
