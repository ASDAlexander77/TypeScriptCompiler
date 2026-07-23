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

    // Accessing an overridden accessor through a base-typed reference used to
    // be a known, pre-existing, non-cross-module-specific gap: get/set
    // accessors were not part of the vtable, so such access resolved
    // statically to the base's own accessor rather than dispatching
    // virtually (see accessor-vtable-dispatch-fix memory). Fixed - this now
    // dispatches to ClampedTemperature's override even through the
    // M.Temperature-typed reference, cross-module.
    const asBase: M.Temperature = t;
    asBase.celsius = -5;
    assert(asBase.celsius == 0);
    assert(asBase.fahrenheit == 32);

    print("done.");
}
