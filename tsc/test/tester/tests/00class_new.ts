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

function main() {
    const square = new Rectangle(10, 10);
    print(square.area); // 100
    assert(square.area == 100.0);
    print("done.");
}
