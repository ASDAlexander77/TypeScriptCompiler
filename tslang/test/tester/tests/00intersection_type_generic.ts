function extend<T, U>(first: T, second: U): T & U {
    let result = <T & U>{ ...first, ...second };
    return result;
}
  
function main()
{
      const x = extend({ a: "hello" }, { b: 42 });
      const s = x.a;
      const n = x.b;

      print (s);
      print (n);

      assert (s === "hello");
      assert (n === 42);

      print("done.");
}