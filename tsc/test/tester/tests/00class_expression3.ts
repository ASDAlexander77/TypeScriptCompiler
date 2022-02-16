let Point = class {
    constructor(public x: number, public y: number) { }
    public length() {
        return (this.x * this.x + this.y * this.y);
    }
};

function main() {
    const p = new Point(3, 4); // p has anonymous class type
    print(p.length());
    print("done.");
}