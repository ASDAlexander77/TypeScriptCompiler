function render(screenWidth, screenHeight) {
    const getPoint = (x, y) => {
        var recenterX = x =>(x - (screenWidth / 2.0)) / 2.0 / screenWidth;
        var recenterY = y => - (y - (screenHeight / 2.0)) / 2.0 / screenHeight;
        return recenterX(x) + recenterY(y);
    }

    const v = getPoint(0, 0);
}

render(8, 6);

print('done');