// Testing a union of null and two or more other types for truth failed to compile: "can't
// instantiate function. '___cast' not all generic types can be identified", after "can't cast
// from 'null' to 'boolean'" reported at an unrelated place. `boolean && (Handle | null)` is such
// a union, so `if (enabled && handle)` failed the same way.

class Handle {
    value: number;
    constructor() {
        this.value = 7;
    }
}

function use(enabled: boolean, handle: Handle | null): number {
    if (enabled && handle) {
        return handle.value;
    }

    return 0;
}

function useReversed(enabled: boolean, handle: Handle | null): number {
    if (handle && enabled) {
        return 1;
    }

    return 0;
}

function truthy(x: boolean | string | null): boolean {
    if (x) {
        return true;
    }

    return false;
}

function notTruthy(x: boolean | number | null): boolean {
    return !x;
}

function andText(e: boolean, s: string | null): number {
    return e && s ? 1 : 0;
}

function main() {
    const h = new Handle();
    assert(use(true, h) == 7, "use(true, h)");
    assert(use(false, h) == 0, "use(false, h)");
    assert(use(true, null) == 0, "use(true, null)");
    assert(useReversed(true, h) == 1, "useReversed(true, h)");
    assert(useReversed(true, null) == 0, "useReversed(true, null)");

    assert(truthy("a"), "truthy(\"a\")");
    assert(!truthy(""), "truthy(\"\")");
    assert(truthy(true), "truthy(true)");
    assert(!truthy(false), "truthy(false)");
    assert(!truthy(null), "truthy(null)");

    assert(notTruthy(null), "!null");
    assert(notTruthy(0), "!0");
    assert(!notTruthy(3), "!3");
    assert(!notTruthy(true), "!true");

    assert(andText(true, "x") == 1, "true && \"x\"");
    assert(andText(true, null) == 0, "true && null");
    assert(andText(false, "x") == 0, "false && \"x\"");

    print("done.");
}
