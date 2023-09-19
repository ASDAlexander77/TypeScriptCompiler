const c = "c";
const d = 10;
// TODO:
//const e = Symbol();

const enum E1 {
  A,
  B,
  C,
}

const enum E2 {
  A = "A",
  B = "B",
  C = "C",
}

type Foo = {
  a: string; // String-like name
  5: string; // Number-like name
  // TODO:
  //[c]: string; // String-like name
  // TODO:
  //[d]: string; // Number-like name
  // TODO:
  //[e]: string; // Symbol-like name
  [E1.A]: string; // Number-like name
  [E2.A]: string; // String-like name
};

type K1 = keyof Foo; // "a" | 5 | "c" | 10 | typeof e | E1.A | E2.A
type K2 = Extract<keyof Foo, string>; // "a" | "c" | E2.A
type K3 = Extract<keyof Foo, number>; // 5 | 10 | E1.A
type K4 = Extract<keyof Foo, symbol>; // typeof e

function main()
{
    print("done.");
}