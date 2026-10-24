// Methods of an ordinary class that call each other before all of them are declared, returning a
// @value class: while the first is discovered the others' results are not known yet, and that
// unresolved result is not a conversion to report (the ray tracer's traceRay/shade/reflection).
@value
class Color {
    constructor(public readonly r: number) {}

    static scale(k: number, v: Color) {
        return new Color(k * v.r);
    }

    static grey = new Color(0.5);
}

class RT {
    private maxDepth = 5;

    private traceRay(x: number, depth: number): Color {
        if (depth > 3) {
            return Color.grey;
        }

        return this.shade(x, depth);
    }

    private shade(x: number, depth: number) {
        return depth >= this.maxDepth ? Color.grey : this.reflect(x, depth);
    }

    private reflect(x: number, depth: number) {
        return Color.scale(x, this.traceRay(x, depth + 1));
    }

    run() {
        return this.traceRay(2, 0).r;
    }
}

function main() {
    assert(new RT().run() == 8);
    print("done.");
}
