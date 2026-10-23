// A record a call returns that borrows its argument (spec 32), held in a `const`: a field read out of
// it reads the borrowed argument, so it cannot come after the argument is overwritten.
class Ray {
    constructor(public x: number) {}
}

interface Hit {
    ray: Ray;
    t: number;
}

function hit(r: Ray): Hit {
    return { ray: r, t: 1 };
}

function main() {
    let r = new Ray(1);
    const h = hit(r);
    r = new Ray(2);
    print(h.ray.x);
}
