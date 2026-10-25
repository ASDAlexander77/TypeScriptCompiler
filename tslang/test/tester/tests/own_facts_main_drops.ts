// own-facts-objects.cmake, the negative: replaceCam, another module's function, overwrites the
// Holder's `cam` its caller's borrow was read from. Its facts (--own-facts) say so, and the borrow
// used after the call is still an error.
import { Cam, Holder, replaceCam } from "./own_facts_lib";

function main() {
    const h = new Holder();
    const c = h.cam;
    replaceCam(h);
    print(c.size());
}
