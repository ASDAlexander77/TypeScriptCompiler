// An arrow with an untyped parameter inside another one with a parameter of the same name, which
// captures a variable of the function around both, failed to compile: "can't instantiate function
// ... not all generic types can be identified". Each untyped parameter makes its arrow generic in
// a type named after the parameter, so the inner `x`'s type had the same name as the outer `x`'s,
// and instantiating the outer arrow bound the inner one's type too.

function render(screenWidth: number, screenHeight: number) {
    const getPoint = (x, y) => {
        const recenterX = x => (x - (screenWidth / 2.0)) / 2.0 / screenWidth;
        const recenterY = y => -(y - (screenHeight / 2.0)) / 2.0 / screenHeight;
        return recenterX(x) + recenterY(y);
    };

    assert(getPoint(512, 384) == 0, "getPoint(512, 384)");
    assert(getPoint(1024, 768) == 0, "getPoint(1024, 768)");
    const v = getPoint(300, 200);
    assert(v - 0.016276 < 0.0001 && v - 0.016276 > -0.0001, "getPoint(300, 200)");
}

function untypedOuter(w, h) {
    const add = (x, y) => {
        const shift = x => x + w;
        return shift(x) + y + h;
    };

    return add(1, 2);
}

function sameNameNoCapture() {
    const outer = (x) => {
        const inner = x => x * 2;
        return inner(x) + 1;
    };

    return outer(5);
}

function main() {
    render(1024, 768);
    assert(untypedOuter(10, 100) == 113, "untypedOuter");
    assert(sameNameNoCapture() == 11, "sameNameNoCapture");
    print("done.");
}
