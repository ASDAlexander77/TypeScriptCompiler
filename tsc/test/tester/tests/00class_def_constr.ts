class Point {
    x = 0;
    y = 0;
}

class PointX {
    x = 1;
}

class PointY extends PointX {
    y = 2;
}

function main() {
    const pt = new Point();
    // Prints 0, 0
    print(`${pt.x}, ${pt.y}`);

    const pt2 = new PointY();
    // Prints 0, 0
    print(`${pt2.x}, ${pt2.y}`);

    print("done.");
}
