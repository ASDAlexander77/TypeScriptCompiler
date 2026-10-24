// @value across a module boundary (export_value_types.ts): the imported class `V` is the exporter's
// type, so `V.plus`, `new V` and the getter work on values from either side. Values come back from
// the library's functions and go into them, built here as literals whose fields are written out of
// declaration order, nested in another; a nested field is read through two levels; a `Hit | null`
// is narrowed and read through `?.`.
import './export_value_types'

function main() {
    let sum: number = 0;
    for (let i = 0; i < 1000; i++) {
        sum += V.plus(mkR(i).a, new V(1)).twice;
    }

    assert(sum == 1001000, "V.plus of an imported value and a local one");

    const r: R = { n: 3, a: new V(4) };
    assert(readR(r) == 43, "a local R read by the library");
    assert(readR(mkR(5)) == 55, "the library's R given back to it");

    const h = mkHit(2);
    assert(h.ray.dir.x == 3 && h.ray.start.x == 2 && h.dist == 4, "a nested field of an imported Hit");

    const local: Hit = { dist: 5, ray: { dir: new V(7), start: new V(6) } };
    assert(readHit(local) == 576, "a local Hit, fields out of order, read by the library");
    assert(local.ray.dir.x == 7 && local.ray.start.x == 6 && local.dist == 5, "the local Hit read here");

    const f = find(3);
    if (f) {
        assert(f.dist == 6 && f.ray.dir.x == 4, "a found Hit");
    } else {
        assert(false, "find(3) found nothing");
    }

    const g = find(-1);
    if (g) {
        assert(false, "find(-1) found something");
    }

    assert(f?.dist == 6, "?. on a found Hit");
    assert(g?.dist === undefined, "?. on a missing Hit");

    print("done.");
}
