// `J | null` given to `I | null`, where J does not extend I, was a plain cast of the union: the value kept
// J's vtable and I read it by its own slot numbers (00interface_cast_unrelated.ts has the cast of J to I
// alone). `C | null` given to an interface (through `I | undefined`) was a class pointer read as an
// interface. Both are the cast of the union's one member now; null stays null.
interface Hit {
    thing: number;
    ray: number;
    dist: number;
}

interface HasDist {
    dist: number;
}

interface A {
    f(): number;
    g(): number;
}

interface B {
    g(): number;
}

class AC implements A {
    base: number = 100;
    f(): number {
        return this.base + 1;
    }
    g(): number {
        return this.base + 2;
    }
}

class HitC implements Hit {
    thing: number = 10;
    ray: number = 20;
    dist: number = 30;
}

function pick(flag: boolean): Hit | null {
    if (flag) {
        const h: Hit = { thing: 1, ray: 2, dist: 3 };
        return h;
    }

    return null;
}

function pickC(flag: boolean): HitC | null {
    if (flag) {
        return new HitC();
    }

    return null;
}

function distOrMinus(d: HasDist | null) {
    return d ? d.dist : -1;
}

function main() {
    // `Hit | null` to `HasDist | null`
    const h = pick(true);
    const d: HasDist | null = h;
    assert(d !== null && d.dist == 3, "a field through J | null");
    assert(distOrMinus(h) == 3, "a field through J | null as an argument");

    const n = pick(false);
    const dn: HasDist | null = n;
    assert(dn === null, "null stays null");
    assert(distOrMinus(n) == -1, "null as an argument");

    // methods, `A | null` to `B | null`
    const a: A | null = new AC();
    const b: B | null = a;
    assert(b !== null && b.g() == 102, "a method through J | null");

    // to `I | undefined`, from an interface and from a class
    const u: HasDist | undefined = h;
    assert(u !== undefined && u.dist == 3, "an interface through I | undefined");
    const c: HasDist | undefined = pickC(true);
    assert(c !== undefined && c.dist == 30, "a class through I | undefined");

    // `<I>` of `J | null`
    const cast = <HasDist>h;
    assert(cast.dist == 3, "<I> of J | null");

    let sum: number = 0;
    for (let i = 0; i < 100; i++) {
        const each: HasDist | null = pick(i % 2 == 0);
        sum += distOrMinus(each);
    }

    assert(sum == 50 * 3 - 50, "casts in a loop");

    print("done.");
}
