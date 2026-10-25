// The program half of own-facts-objects.cmake: a borrow of `this.cam` held across calls of
// another module's methods. With --own-facts=<own_facts_lib's object> -mm=own knows Cam.reset
// overwrites only a Cam's `position` and Cam.size nothing, so the borrow survives both; without it
// either call may destroy anything, and the borrow is an error. `shifted`'s result is fresh, so it
// may replace the array it was given.
import { Cam, shifted } from "./own_facts_lib";

function churn() {
    const keep: Cam[] = [];
    for (let i = 0; i < 2000; i++) {
        const c = new Cam();
        c.reset(77777.0);
        keep.push(c);
    }

    return keep.length;
}

class Pass {
    private cam: Cam = new Cam();
    private offset: number[] = [1.0, 5.0];

    run(): number {
        this.cam.reset(3.0);
        churn();
        this.offset = shifted(this.offset, 1.0);
        churn();
        return this.cam.position[0] + this.cam.position[1] + this.cam.size() + this.offset[0];
    }
}

function main() {
    const p = new Pass();
    print(p.run());   // 11
    print("done.");
}
