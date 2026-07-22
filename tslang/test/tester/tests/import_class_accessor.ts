import './export_class_accessor'

class ClampedTemperature extends M.Temperature {
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

    // NOTE: accessing an overridden accessor through a base-typed reference
    // (`const asBase: M.Temperature = t; asBase.celsius`) is a known,
    // pre-existing, non-cross-module-specific gap: get/set accessors are not
    // part of the vtable, so such access resolves statically to the base's
    // own accessor rather than dispatching virtually. Not exercised here -
    // this test only covers the super-accessor-call crash fix.

    print("done.");
}
