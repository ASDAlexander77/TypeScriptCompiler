// A field whose written type names a class declared further down, with an initializer:
// `particles: Particle[] = []` was typed by the initializer (`any[]`) because Particle was not
// known yet, and its elements were `any` from then on.
function main() {
    const g = new G();
    g.particles.push(new Particle(3.0));
    const p: Particle = g.particles[0];
    assert(p.duration == 3.0, "element read");
    assert(g.particles[0].duration == 3.0, "element member");
    assert(g.first() == 3.0, "method");

    print("done.");
}

class G {
    particles: Particle[] = [];

    first() {
        return this.particles[0].duration;
    }
}

class Particle {
    constructor(public duration: number) {}
}
