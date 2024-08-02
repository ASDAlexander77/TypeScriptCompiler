class S<T> {
  static test<V>(this: S<T>, t: T) {
    if (t > 0) return <V>10;
    return <V>1;
  }

  static test2<V>(t: T) {
    if (t > 0) return <V>11;
    return <V>1;
  }

}

function main() {

  const s = new S<int>();  

  const r = s.test<i32>(10);
  print(r);
  assert(r == 10);

  const r2 = s.test2<i32>(10);
  print(r2);

  assert(r2 == 11);
  print("done.");
}
