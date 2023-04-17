class Array1 {
    static [Symbol.hasInstance](instance) {
      if (typeof instance === "string")
      print("str: ", instance);
      if (typeof instance === "number")
      print("num: ", instance);
      return false;
    }
  }
  
  interface IObj
  {
      [Symbol.hasInstance]: (v: any) => boolean;
  }
  
  function main()
  {
      Array1[Symbol.hasInstance]("hello");
      Array1[Symbol.hasInstance](<number>10);
  
      const obj = {
          [Symbol.hasInstance]: (instance) => {
              if (typeof instance === "string")
              print("obj: str: ", instance);
              if (typeof instance === "number")
              print("obj: num: ", instance);
              return true;
          }
      };
  
      obj[Symbol.hasInstance](<number>20);
  
      const iobj: IObj = obj;
      iobj[Symbol.hasInstance](<number>30);
  
      print("done.");
  }