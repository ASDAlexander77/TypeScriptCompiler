// TSLANG 190: under -mm=rc a callee with no body in this module (another module's) was taken to write nothing,
// so `clobber(arr[0], h)` - which sets `h.items[0]`, the same element - got no hold of the argument, and the
// store freed the box `clobber` then read (the program spun in a corrupted heap). A callee with no body is
// taken to write any place of an object it is given.
import { Box, Holder, clobber } from "./export_bodiless_writes";

function main() {
    let total = 0;
    for (let i = 0; i < 1000; i++) {
        const arr: Box[] = [new Box("old" + i)];
        const h = new Holder(arr);
        const s = clobber(arr[0], h);
        assert(s == "old" + i, "the argument read before the call");
        assert(arr[0].s == "new1", "the element given a new value");
        total += s.length;
    }

    assert(total == 5890, "every call");
    print("done.");
}
