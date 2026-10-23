// Shared<I> of an object literal (spec 26): only a value made in the global's initializer moves into
// the handle. `b`'s initializer gives `new Shared` the payload `a` holds - a second owner of `a`'s
// block, which -mm=own does not count: rejected.
interface S { f: (x: number) => number; roughness: number; }
let a = new Shared<S>({ f: function (x: number) { return x * 2; }, roughness: 2 });
let b = new Shared<S>(a.value);
function main() {
    print(a.value.roughness, b.value.roughness);
}
