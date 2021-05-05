function f1() {
    print("Hello World!");
}

function run(f: () => void) {
    f();
}

function main() {
    const x = f1;
    x();
    run(x);

	(function () {
		print("Hello World!");
	})();

    nested();
    init_method();
    arrow_func();
}

function init_method() {                                                 
    let greeting = function() {
        print("Hello TypeScript!");
    };
    
    greeting();
}

function arrow_func() {                                                 

    let sum = (x: number, y: number): number => {
        return x + y;
    }
    
    sum(10, 20);
    
}

function nested() {                                                 
    function _x() {                      
        print(1);                   
        return 1;                         
    }                                     
    function _y() {                       
        print(2);                   
        return 2;                         
    }                                     

    print(_x() || _x() == 0 ? _x() : _y());     
    print(!_x() && _x() != 0 ? _x() : _y());     
}

