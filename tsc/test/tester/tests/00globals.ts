let glb1 = 1;
const glb2 = 2;

function main() {    
    assert(glb1 == 1, "Failed. glb1");
    assert(glb2 == 2, "Failed. glb2");
    glb1 = glb1 + 1;
    assert(glb1 == 2, "Failed. glb1++");
}
