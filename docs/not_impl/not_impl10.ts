type Maybe<T> = T | void;

function isDefined<T>(x: Maybe<T>): x is T {
  return x !== undefined && x !== null;
}

function test1(x: Maybe<string>) {
  let x2 = isDefined(x) ? x : "Undefined"; // string
}

function main()
{
}