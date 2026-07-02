class Rectangle {
    height: number;
    width: number;

    constructor(height: number, width: number) {
        this.height = height;
        this.width = width;
    }
    // Getter
    get area() {
        return this.calcArea();
    }
    // Method
    calcArea() {
        return this.height * this.width;
    }
}
class Vector {
    constructor(public x: number,
        public y: number,
        public z: number) {
    }

    static def = new Vector(0, 0, 0);

    times() { return Vector.def; }
}

function main() {
    const square = new Rectangle(10, 10);
    print(square.area); // 100
    assert(square.area == 100.0);

    // if it is compiled it is fine
    const s = new Vector(1, 2, 3);
    print(s.x, s.y, s.z);

    print("done.");
}
