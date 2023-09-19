let glb1 = 0;

namespace fooX.bar.baz {
    export function b() {
        glb1++;
    }
}

import bz = fooX.bar.baz;
function testImports() {
    glb1 = 0;
    bz.b();
    assert(glb1 == 1, "imports");
}

function main() {
    testImports();
    print("done.");
}
