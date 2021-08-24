class IntIter {
    constructor(private i = 0) {}
    next() {
        type retType = [value: typeof this.i, done: boolean];
        if (this.i < 10) {
            return <retType>[this.i++, false];
        }

        return <retType>[this.i, true];
    }
}

function main1() {
    let it = new IntIter();

    let count = 0;
    for (const o of it) {
        count++;
        print(o);
    }

    assert(count == 10);

    count = 0;
    for (const o of "Hello") {
        count++;
        print(o);
    }

    assert(count == 5);
}

// ver 2
function foo2()
{
	return {
		step: 0,
		next() {
			switch (this.step++)
			{
			  	//yield 1;
				case 0: 
					return { value: 1, done: false; };
			  	//yield 2;
				case 1: 
					return { value: 2, done: false; };
			  	//yield 3;
				case 2:
					return { value: 3, done: false; };
				default:
					return { value: 3, done: true; };					
			}
		}
	}

};

function main2() {

    let count = 0;
    for (const o of foo2())
    {
    	print (o);
        count++;
    }

    assert(count == 3);
}

// ver 3
function foo3()
{
    let step = 0;
	return {
		next() {
			switch (step++)
			{
			  	//yield 1;
				case 0: 
					return { value: 1, done: false; };
			  	//yield 2;
				case 1: 
					return { value: 2, done: false; };
			  	//yield 3;
				case 2:
					return { value: 3, done: false; };
				default:
					return { value: 3, done: true; };					
			}
		}
	}

};

function main3() {

    let count = 0;
    for (const o of foo3())
    {
    	print (o);
        count++;
    }

    assert(count == 3);    
}

function main()
{
    main1();
    main2();
    main3();
    print("done.");
}