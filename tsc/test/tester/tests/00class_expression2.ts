function getClass() {
    let Point = class {
        constructor(public x: number, public y: number) { }
        public length() {
            return (this.x * this.x + this.y * this.y);
        }
    };

    return Point;
}

function main() {
    const PointClass = getClass();
    const p = new PointClass(3, 4); // p has anonymous class type
    print(p.length());

    assert(p.length() == 25);

    print("done.");
}