interface Shape {
    color: string;
  }
   
  interface PenStroke {
    penWidth: number;
  }
   
  interface Square extends Shape, PenStroke {
    sideLength: number;
  }
  
  function main()
  { 
      let square = {} as Square;
      square.color = "blue";
      square.sideLength = 10;
      square.penWidth = 5.0;
  
      print(`${square.color}, ${square.sideLength}, ${square.penWidth}`);

      assert(square.color == "blue");
      assert(square.sideLength == 10);
      assert(square.penWidth == 5.0);
  
      print("done.");
  }