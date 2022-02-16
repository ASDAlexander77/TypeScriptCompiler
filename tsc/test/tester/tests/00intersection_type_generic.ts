function extend<T, U>(first: T, second: U): T & U {
    let result = <T & U>{};
    return result;
  }
  
  function main()
  {
      const x = extend({ a: "hello" }, { b: 42 });
      const s = x.a;
      const n = x.b;

      print("done.");
  }