function f3() {
  function f<X, Y>(x: X, y: Y) {
    class C {
      public x: X;
      public y: Y;
    }
    return C;
  }
  let C = f(10, "hello");
  let v = new C();
  let x = v.x; // number
  let y = v.y; // string
}

function main()
{
	f3();
}