class Point {
    x: number;
    y: number;
}

class Line {
    constructor(public start: Point, public end: Point) { }
}

const l = new Line({ x: 0, y: 1 }, { x: 1.0, y: 2.0 });

print (l.start.x, l.start.y, l.end.x, l.end.y);

assert(l.start.x == 0 && l.start.y == 1 && l.end.x == 1.0 && l.end.y == 2.0);

print("done.");

