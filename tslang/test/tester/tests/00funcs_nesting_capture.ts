function render(screenWidth, screenHeight) {
    const getPoint = (x, y) => {
        const recenterX = x1 => (x1 - (screenWidth / 2.0)) / 2.0 / screenWidth;
        const recenterY = y1 => -(y1 - (screenHeight / 2.0)) / 2.0 / screenHeight;
        return recenterX(x) + recenterY(y);
    }

    const v = getPoint(512, 384);
    print(v);
    assert(v === 0);

    const v2 = getPoint(1024, 768);
    print(v2);
    assert(v2 === 0);

    const v3 = getPoint(300, 200);
    print(v3);
    assert((v3 - 0.016276) < 0.0001);

}

render(1024, 768);

print("done.");