// A union is stored as one byte buffer. Each member must read back exactly what it was
// stored with, even when its fields sit where another member's struct has padding.
//
// A tagged union lowers to { tag, S }, where S is the type of its largest member. If S is that
// member's struct type, a smaller member's field that falls in S's padding is not part of the
// loaded value, so it is lost the next time the union is copied as a value.

// Three is { i1, i32, i32 } (12 bytes, 4-aligned); Big is { i1, double } (16 bytes, 8-aligned) at
// both x64 and i686, so Big is the storage. Three.a (offset 4) lies in Big's padding (offsets 1-7)
// at x64 and at i686. Three.b (offset 8) overlaps Big.v and survives.
type Three = { flag: boolean; a: s32; b: s32 };
type Big = { flag: boolean; v: number };
type U1 = Three | Big;

function makeThree(): U1 { return { flag: true, a: 7, b: 9 }; }
function makeBig(): U1 { return { flag: false, v: 1.5 }; }

// At x64 every member is pointer-and-double sized, with no padding, and Success is the storage.
// At i686 Success is { ptr, pad4, { ptr, pad4, double, ptr } } (32 bytes) and Failed is
// { ptr, pad4, double }: the upper half of Failed.code (offsets 12-15) lies in the padding after
// Success.response.title, so `code` reads garbage at i686.
type Loading = { state: string };
type Failed = { state: string; code: number };
type Success = { state: string; response: { title: string; duration: number; summary: string } };
type Net = Loading | Failed | Success;

function describe(s: Net): string {
    if (s.state == "failed") return `code ${(<Failed><any>s).code}`;
    if (s.state == "success") return `title ${(<Success><any>s).response.title}`;
    return "loading";
}

function main() {
    const t = makeThree();
    const t2 = t;                       // copy the union as a value
    const three = <Three><any>t2;
    assert(three.flag, "three.flag");
    assert(three.a == 7, "three.a");
    assert(three.b == 9, "three.b");

    const b = makeBig();
    const b2 = b;
    assert(!(<Big><any>b2).flag, "big.flag");
    assert((<Big><any>b2).v == 1.5, "big.v");

    const f: Net = { state: "failed", code: 1.0 };
    const f2 = f;
    assert(describe(f2) == "code 1", "describe(failed)");
    assert(describe({ state: "success", response: { title: "t", duration: 2.5, summary: "s" } }) == "title t", "describe(success)");
    assert(describe({ state: "loading" }) == "loading", "describe(loading)");

    print("done.");
}
