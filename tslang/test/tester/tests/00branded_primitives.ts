// A branded primitive (`string & { __brand: "UserId" }`): a primitive and object types that only tag it at compile
// time. Its values are the primitive's, as in TypeScript; it was `never` ("'Never' can't be used as parameter type").
type UserId = string & { __brand: "UserId" };
type Meters = number & { readonly _unit: "m" };

function mk(s: string): UserId { return s as UserId; }
function twice(m: Meters): number { return m * 2; }
function next(x: { _foo: number } & number) { return x + 1; }

function main() {
    const id = mk("abc");
    assert(id.length == 3 && id == "abc");
    assert(twice(5 as Meters) == 10);
    assert(next(1 as any) == 2);
    const ids: UserId[] = [mk("a"), mk("b")];
    assert(ids.length == 2 && ids[1] == "b");
    print("done.");
}
