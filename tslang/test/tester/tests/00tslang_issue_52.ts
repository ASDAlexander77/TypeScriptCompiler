// TSLANG_ISSUES 52 (-mm=own), from native_ts_graphics' core/nri_donut_textures.ts: a value of the
// function's own pushed into an array that is a field, then used - `this.textures.push(texture);
// this.views.push(texture.createView(...))`. After the push the value is the array's last element, a
// borrow of it; pushing into other arrays, or more into the same one, keeps it.

class Tex {
    id: int = 1;
    view(): int {
        return this.id + 100;
    }
}

@final
class Device {
    load(fail: boolean): Tex | null {
        return fail ? null : new Tex();
    }
}

export class Textures {
    private textures: Tex[] = [];
    private srgb: boolean[] = [];
    private views: int[] = [];

    load(device: Device, n: int): boolean {
        for (let i = 0; i < n; i++) {
            const texture = device.load(false);
            if (!texture) {
                return false;
            }

            this.textures.push(texture);
            this.srgb.push(i > 0);
            this.views.push(texture.view());
        }

        return true;
    }

    // a `new` object, the same array pushed into again before the use
    twice(): int {
        const t = new Tex();
        t.id = 7;
        this.textures.push(t);
        this.textures.push(new Tex());
        return t.id + this.textures.length;
    }

    sum(): int {
        let total = 0;
        for (const v of this.views) {
            total += v;
        }

        return total;
    }
}

function main() {
    const t = new Textures();
    assert(t.load(new Device(), 3));
    print(t.sum());
    assert(t.sum() == 303);
    assert(t.twice() == 12);
    print("done.");
}
