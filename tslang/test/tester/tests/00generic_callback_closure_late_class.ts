// A closure (it reads `this`) given to a generic function in a class method, the array's element a
// class declared further down. While the method's return type was discovered, the arrow's type was
// rebuilt with its captures as its first input, and inference paired them with `T`: `T` was
// `Late | captures`. Here that failed every time ("cast from array type ... to array type
// '!ts.array<!ts.union<... !ts.ref<!ts.tuple<{"this",...'"); with the default library's generic
// `Array.filter` the union's members came in an order of their types' addresses, so tslang crashed
// in one run in four (0xC0000005, native_ts_graphics' ray_tracing_extended_scene.ts).
let seed = 1.0;
function rnd() {
    seed = (seed * 16807.0) % 2147483647.0;
    return seed / 2147483647.0;
}

function keep<T>(items: T[], f: (v: T, i: number) => boolean): T[] {
    const kept: T[] = [];
    for (let i = 0; i < items.length; i++) {
        if (f(items[i], i)) kept.push(items[i]);
    }

    return kept;
}

class Gen {
    particles: Late[] = [];
    lifetime = 2.0;

    update() {
        const remaining = keep(this.particles, (particle) => !(particle.duration > rnd() * this.lifetime));
        this.particles = remaining;
        return remaining.length;
    }

    firstDuration() {
        return this.particles[0].duration;
    }

    longer(min: number) {
        return keep(this.particles, (particle) => particle.duration >= min * this.lifetime).length;
    }
}

class Late {
    constructor(public duration: number) {}
}

function main() {
    const g = new Gen();
    g.particles.push(new Late(0.0), new Late(100.0), new Late(0.0));
    assert(g.update() == 2, "the long-lived one is kept out");
    assert(g.particles.length == 2 && g.firstDuration() == 0.0);
    g.particles.push(new Late(7.0));
    assert(g.longer(1.0) == 1, "a second closure over this");
    print("done.");
}
