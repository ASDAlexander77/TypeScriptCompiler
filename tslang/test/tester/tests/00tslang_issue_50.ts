// TSLANG_ISSUES 50 (-mm=own), from native_ts_graphics' core/nri_donut_blit.ts: a value of the
// function's own (a call's result, a `new`) kept in a `const`, moved into a field, and then used -
// tested, passed to a call, read in a `?:`. After the move the `const` is what the field holds, a
// borrow of it; the calls between write no field of `this`.

class Layout {
    id: int = 1;
}

@final
class Device {
    made = 0;
    make(fail: boolean): Layout | null {
        this.made++;
        return fail ? null : new Layout();
    }
    use(l: Layout): int {
        return l.id + 10;
    }
}

export class Blit {
    private layout: Layout | null = null;
    private kept: Layout | null = null;
    n = 0;

    init(device: Device, fail: boolean): boolean {
        const layout = device.make(fail);
        this.layout = layout;
        if (!layout) {
            return false;
        }

        device.make(true);
        this.n = layout ? device.use(layout) : -1;
        return true;
    }

    // a `new` object moved into a field that may be null: read back narrowed to the object
    keep(device: Device) {
        const l = new Layout();
        l.id = 5;
        this.kept = l;
        this.n += device.use(l) + l.id;
    }

    // the store in an `if`
    nested(device: Device, ok: boolean): int {
        if (ok) {
            const layout = device.make(this.n < 0);
            this.layout = layout;
            if (!layout) {
                return -1;
            }

            return device.use(layout);
        }

        return 0;
    }
}

// a string is copied into a field and a Shared<T> handle counted, not moved: the function's own
// value stays usable after a call that overwrites the field
function mkText(): string {
    return "a" + "b";
}

class Copies {
    name: string = "";
    shared: Shared<Layout> | null = null;
    clear() {
        this.name = "z";
        this.shared = null;
    }
    run(): string {
        const s = mkText();
        this.name = s;
        const h = new Shared<Layout>(new Layout());
        this.shared = h;
        this.clear();
        return `${s}${h.value.id}`;
    }
}

function main() {
    assert(new Copies().run() == "ab1");
    const device = new Device();
    const b = new Blit();
    assert(b.init(device, false));
    assert(b.n == 11);
    assert(!b.init(device, true));
    b.keep(device);
    print(b.n, device.made);
    assert(b.n == 11 + 15 + 5 && device.made == 3);
    assert(b.nested(device, true) == 11 && b.nested(device, false) == 0);
    print("done.");
}
