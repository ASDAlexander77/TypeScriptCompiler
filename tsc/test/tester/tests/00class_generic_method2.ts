class S<T> {

  test<V>(t: T) {
    if (t > 0) return <V>11;
    return <V>1;
  }

}

function main() {

  const s = new S<int>();  

  const r = s.test<i32>(10);
  print(r);
  assert(r == 11);

  print("done.");
}
