// The library half of own-facts-objects.cmake: compiled into an object of its own under -mm=own,
// whose __tsownfacts_ says what its exported functions destroy (SHARED_LIB_OWN_OBJECT_FACTS).
// Cam.reset overwrites only its own `position`; Cam.size destroys nothing; replaceCam overwrites
// a Holder's `cam`, the place an importer's borrow is read from in own_facts_main_drops.ts.
@final
export class Cam {
    position: number[] = [0.0, 0.0];

    reset(x: number): void {
        this.position = [x, x + 1.0];
    }

    size(): number {
        return this.position.length;
    }
}

export class Holder {
    cam: Cam = new Cam();
}

export function replaceCam(h: Holder): void {
    h.cam = new Cam();
}

// returns an array its caller owns, borrowing neither argument: the facts say so (`fresh`), and an
// importer may store it over the place it read `a` from
export function shifted(a: number[], d: number): number[] {
    return [a[0] + d, a[1] + d];
}
