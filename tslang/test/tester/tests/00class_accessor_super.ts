class Temperature {
    protected _celsius: number = 0;

    get celsius(): number {
        return this._celsius;
    }

    set celsius(v: number) {
        this._celsius = v;
    }

    get fahrenheit(): number {
        return this._celsius * 9 / 5 + 32;
    }
}

// overriding an accessor and calling back into the base implementation via
// `super.<accessor>` / `super.<accessor> = value` used to crash MLIR
// verification: the setter call's `this` operand was left as a raw by-value
// ClassStorageType struct instead of being materialized to a pointer (the
// same repair ordinary `super.method()` calls already got via
// getThisRefOfClass), so `'llvm.call' op operand type mismatch` was thrown
// the first time this combination was exercised (found via the cross-module
// accessor test, but the bug itself is general, not cross-module-specific).
class ClampedTemperature extends Temperature {
    get celsius(): number {
        return super.celsius;
    }

    set celsius(v: number) {
        super.celsius = v < 0 ? 0 : v;
    }
}

function main() {
    const t = new ClampedTemperature();
    t.celsius = -10;
    assert(t.celsius == 0);
    assert(t.fahrenheit == 32);

    t.celsius = 100;
    assert(t.celsius == 100);
    assert(t.fahrenheit == 212);

    print("done.");
}
