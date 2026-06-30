class Person {
    static accessor sname: string;
    accessor name = "no value";
    constructor(name: string) {
        this.name = name;
    }
}

function main()
{
	const p = new Person("hello");	
	print(p.name);
	p.name = "hello2";
	print(p.name);

	Person.sname = "sname1";	

	print(Person.sname);

	print("done.");
}