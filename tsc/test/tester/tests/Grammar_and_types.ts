function main()
{
    // Basics
    let Früh = "foobar";

    // Comments
    // a one line comment

    /* this is a longer,
    * multi-line comment
    */

    /* You can't, however, /* nest comments */

    // Evaluating variables

    var a;
    console.log('The value of a is ' + a); // The value of a is undefined
    
    console.log('The value of b is ' + b); // The value of b is undefined
    var b;
    // This one may puzzle you until you read 'Variable hoisting' below
    
    console.log('The value of c is ' + c); // Uncaught ReferenceError: c is not defined
    
    let x;
    console.log('The value of x is ' + x); // The value of x is undefined
    
    console.log('The value of y is ' + y); // Uncaught ReferenceError: y is not defined
    let y; 
    
    var input;
    if (input === undefined) {
        doThis();
    } else {
        doThat();
    }

    var myArray = [];
    if (!myArray[0]) myFunction();
    
    a + 2;  // Evaluates to NaN

    var n = null;
    console.log(n * 32); // Will log 0 to the console
    
    // Variable scope
    if (true) {
        var x = 5;
    }
    console.log(x);  // x is 5    

    // Variable hoisting

    /**
     * Example 1
     */
    console.log(x === undefined); // true
    var x = 3;
    
    /**
     * Example 2
     */
    // will return a value of undefined
    var myvar = 'my value';
    
    (function() {
        var myvar = 'local value';
        console.log(myvar); // undefined
    })();    

    // Function hoisting
    foo(); // "bar"

    function foo() {
      console.log('bar');
    }
    
    /* Function expression */   
    var baz = function() {
      console.log('bar2');
    };    

    baz();

    // Constants

    const PI = 3.14;

    const MY_OBJECT = {'key': 'value'};
    MY_OBJECT.key = 'otherValue';
    
    const MY_ARRAY = ['HTML','CSS'];
    // TODO: //MY_ARRAY.push('JAVASCRIPT');
    console.log(MY_ARRAY); //logs ['HTML','CSS','JAVASCRIPT'];    

    // Data type conversion

    var answer = 42;

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

    var sales = 'Toyota';

    function carTypes(name) {
      if (name === 'Honda') {
        return name;
      } else {
        return "Sorry, we don't sell " + name + ".";
      }
    }
    
    var car = { myCar: 'Saturn', getCar: /* TODO: // carTypes('Honda') */ 'Honda', special: sales };
    
    console.log(car.myCar);   // Saturn
    console.log(car.getCar);  // Honda
    console.log(car.special); // Toyota    

    var car2 = { manyCars: {a: 'Saab', b: 'Jeep'}, 7: 'Mazda' };

    console.log(car2.manyCars.b); // Jeep
    console.log(car2[7]); // Mazda    

    var unusualPropertyNames = {
        '': 'An empty string',
        '!': 'Bang!'
    }    

    console.log(unusualPropertyNames['']);  // An empty string
    console.log(unusualPropertyNames['!']); // Bang!    

    // RegExp literals

    var re = /ab+c/;

    // String literals

    'foo';
    "bar";
    '1234';
    'one line \n another line';
    "John's cat";

    console.log("John's cat".length);

    // Basic literal string creation
    `In JavaScript '\n' is a line-feed.`;

    // Multiline strings
    `In JavaScript, template strings can run
    over multiple lines, but double and single
    quoted strings cannot.`;

    // String interpolation
    var name = 'Bob', time = 'today';
    `Hello ${name}, how are you ${time}?`;

    let myTag = (str, name, age) => `${str[0]}${name}${str[1]}${age}${str[2]}`;
    // TODO: // let [name, age] = ['Mika', 28];
    myTag`Participant "${ name }" is ${ age } years old.`;
    // Participant "Mika" is 28 years old.    

    // Using special characters in strings

    'one line \n another line';

    // Escaping characters

    var quote = "He read \"The Cremation of Sam McGee\" by R.W. Service.";
    console.log(quote);
    
    var home = 'c:\\temp';

    var str = 'this string \
    is broken \
    across multiple \
    lines.'
    console.log(str);   // this string is broken across multiple lines.

    var poem =
    'Roses are red,\n\
    Violets are blue.\n\
    Sugar is sweet,\n\
    and so is foo.'    

    var poem2015 =
    `Roses are red,
    Violets are blue.
    Sugar is sweet,
    and so is foo.`     
}

function doThis() {}
function doThat() {}
function myFunction() {}