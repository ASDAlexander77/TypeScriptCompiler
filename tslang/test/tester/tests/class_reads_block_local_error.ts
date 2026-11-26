// A class declared in a block of the module's code, its member reading the block's local: the error a class in a
// function gets, not "'ts.Load' op using value defined outside the region" (TSLANG 197).
for (let i = 0; i < 3; i++) {
    class C {
        f = i;
    }
}

function main() {}
