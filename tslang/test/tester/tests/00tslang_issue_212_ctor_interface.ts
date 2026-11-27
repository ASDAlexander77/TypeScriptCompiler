// TSLANG 212 (139): a class given to an interface of a construct signature only (`interface PtCtor { new (x: number):
// Pt }`) - "the class 'P' itself can't be given as interface 'PtCtor'". It is made over an object of one field, `new`,
// which calls the class's constructor and returns its object as the signature's result. Not under -mm=own, where a
// constructor keeping its parameter (R) and Constructor<T> stay refused (00tslang_issue_213.ts runs the rest there).
interface Pt { x: number; }
interface PtCtor { new (x: number): Pt; }
class P implements Pt { s: string; constructor(public x: number) { this.s = "s" + x; } }
class Q implements Pt { constructor(public x: number) { } }

function make(c: PtCtor, x: number): Pt { return new c(x); }

class R { constructor(public r: string) { } }
interface RCtor { new (r: string): R; }

// a generic one (it was "the class 'A' itself can't be given as interface 'Constructor<A>'", TSLANG 204)
interface Constructor<T> { new (...args: any[]): T; }
class Plain { a = 1; }

function makeR(c: RCtor): R { return new c("r"); }

function main() {
    let total = 0;
    for (let i = 0; i < 3; i++) {
        total += make(P, i).x + make(Q, 10).x;
    }

    assert(total == 33, "constructor interface");

    const ctor: PtCtor = P;
    assert(new ctor(4).x == 4, "constructor interface held");
    assert(makeR(R).r == "r", "returning the class itself");

    const c: Constructor<Plain> = Plain;
    assert(new c().a == 1, "a generic constructor interface");
    print("done.");
}
