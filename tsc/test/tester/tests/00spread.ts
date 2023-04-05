function sum(x = 0, y = 0, z = 0) {
    return x + y + z;
  }
  
  function main()
  {
      const numbers = [1, 2, 3];
      const r = sum(...numbers);
      print(r);
      assert (r === 9);
      print ("done.");
  }