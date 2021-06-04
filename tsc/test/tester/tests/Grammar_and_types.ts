function main()
{
    // Basics
    let foo = "foobar";

    // Comments
    // a one line comment

    /* this is a longer,
    * multi-line comment
    */

    /* You can't, however, /* nest comments */

    // Evaluating variables

    let input = 0;
    if (input === undefined) {
        doThis();
    } else {
        doThat();
    }

    let myArray = [1];
    if (!myArray[0]) myFunction();
    
    let a = 0;
    a + 2;  // Evaluates to NaN

    let n : string = null;
    print(n * 32); // Will log 0 to the console
    
    // Variable scope
    let x = 0;
    if (true) {
        x = 5;
    }
    print(x);  // x is 5    

    // Variable hoisting

    /**
     * Example 1
     */
    print(x === undefined); // false
    x = 3;
    
    /**
     * Example 2
     */
    // will return a value of undefined
    let mylet = 'my value';
    
    (function() {
        let mylet = 'local value';
        print(myvar); // undefined
    })();    

    // Function hoisting
    foo(); // "bar"

    function foo() {
      print('bar');
    }
    
    /* Function expression */   
    let baz = function() {
      print('bar2');
    };    

    baz();

    // Constants

    const PI = 3.14;

    let MY_OBJECT = {'key': 'value'};
    MY_OBJECT.key = 'otherValue';
    
    const MY_ARRAY = ['HTML','CSS'];
    // TODO: //MY_ARRAY.push('JAVASCRIPT');
    print(MY_ARRAY); //logs ['HTML','CSS','JAVASCRIPT'];    

    // Data type conversion

    let answer = 42;

    // TODO?: // answer = 'Thanks for all the fish...';

    // Numbers and the '+' operator

    const x_ = 'The answer is ' + 42 // "The answer is 42"
    const y_ = 42 + ' is the answer' // "42 is the answer"
    
    '37' - 7 // 30
    '37' + 7 // "377"    

    // Converting strings to numbers

    // TODO: parseInt('101', 2) // 5
    parseInt('101')

    '1.1' + '1.1' // '1.11.1'
    (+'1.1') + (+'1.1') // 2.2

    // Literals

    // Array literals

    let coffees = ['French Roast', 'Colombian', 'Kona'];
    let fish = ['Lion', , 'Angel'];

    let myList1 = ['home', , 'school', ];
    let myList2 = [ ,'home', , 'school'];
    let myList3 = ['home', , 'school', , ];

    // Boolean literals

    true;
    false;

    // Numeric literals

    0, 117, -345, 123456789123456789n;             //(decimal, base 10)
    015, 0001, -0o77, 0o777777777777n;             //(octal, base 8)
    0x1123, 0x00111, -0xF1A7, 0x123456789ABCDEFn;  //(hexadecimal, "hex" or base 16)
    0b11, 0b0011, -0b11, 0b11101001010101010101n;  //(binary, base 2)

    // Floating-point literals

    3.1415926;
    -.123456789;
    -3.1E+12;
    .1e-23;

    // Object literals

    const sales = 'Toyota';

    function carTypes(name:string) {
      if (name === 'Honda') {
        return name;
      } else {
        return "Sorry, we don't sell " + name + ".";
      }
    }
    
    let car = { myCar: 'Saturn', getCar: /* TODO: // carTypes('Honda') */ 'Honda', special: sales };
    
    print(car.myCar);   // Saturn
    print(car.getCar);  // Honda
    print(car.special); // Toyota    

    let car2 = { manyCars: {a: 'Saab', b: 'Jeep'}, 7: 'Mazda' };

    print(car2.manyCars.b); // Jeep
    print(car2[7]); // Mazda    

    let unusualPropertyNames = {
        '': 'An empty string',
        '!': 'Bang!'
    }    

    print(unusualPropertyNames['']);  // An empty string
    print(unusualPropertyNames['!']); // Bang!    

    // RegExp literals

    // TODO: // let re = /ab+c/;

    // String literals

    'foo';
    "bar";
    '1234';
    'one line \n another line';
    "John's cat";

    print("John's cat".length);

    // Basic literal string creation
    `In JavaScript '\n' is a line-feed.`;

    // Multiline strings
    `In JavaScript, template strings can run
    over multiple lines, but double and single
    quoted strings cannot.`;

    // String interpolation
    let name = 'Bob', time = 'today';
    `Hello ${name}, how are you ${time}?`;

    let myTag = (str:string[], name:string, age:number) => `${str[0]}${name}${str[1]}${age}${str[2]}`;
    let [name, age] = ['Mika', 28];
    myTag`Participant "${ name }" is ${ age } years old.`;
    // Participant "Mika" is 28 years old.    

    // Using special characters in strings

    'one line \n another line';

    // Escaping characters

    let quote = "He read \"The Cremation of Sam McGee\" by R.W. Service.";
    print(quote);
    
    let home = 'c:\\temp';

    let str = 'this string \
    is broken \
    across multiple \
    lines.'
    print(str);   // this string is broken across multiple lines.

    let poem =
    'Roses are red,\n\
    Violets are blue.\n\
    Sugar is sweet,\n\
    and so is foo.'    

    let poem2015 =
    `Roses are red,
    Violets are blue.
    Sugar is sweet,
    and so is foo.`     
}

function doThis() {}
function doThat() {}
function myFunction() {}