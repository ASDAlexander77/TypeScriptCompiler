// Benchmark (not registered with ctest): three-component vector temporaries in a tight loop.
class Vec3 {
    x: number;
    y: number;
    z: number;
    constructor(x: number, y: number, z: number) {
        this.x = x;
        this.y = y;
        this.z = z;
    }
}

function dot(a: Vec3, b: Vec3) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

function run(n: number) {
    let t: number = 0;
    for (let i = 0; i < n; i++) {
        const a = new Vec3(i, i + 1, i + 2);
        const b = new Vec3(1, 2, 3);
        const d = new Vec3(a.x - b.x, a.y - b.y, a.z - b.z);
        t += dot(d, d) - dot(a, b);
    }
    return t;
}

function main() {
    const t = run(5000000);
    print(t);
    print("done.");
}
