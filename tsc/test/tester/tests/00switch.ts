function test1() {
    for (let a = 0; a < 7; a++) {
        print("case ", a);
        switch (a) {
            case 1:
                print("Hello 1! fall through");
            case 2:
            case 3:
                print("Hello 2!");
                print("and Hello 3!");
                break;
            case 4:
                print("Hello 4!");
                break;
            case 5:
                print("Hello 5!");
                break;
            default:
                print("default");
                break;
        }
    }
}

function test2() {
    let a = 10;
    switch (a) {
        default:
            print("default 2");
            break;
    }
}

function test3() {
    let a = 10.5
    switch (a) {
        case 10.5:
            print("cool. 10.5");
            break;
    }
}

function test4()
{
		let a = 10;
	        switch (a) 
		{                                            
		    default: {
        	        print("default 2");                          
			break;
		   }                                          
	        }
}

function main() {
    test1();
    test2();
    test3();
    test4();
}