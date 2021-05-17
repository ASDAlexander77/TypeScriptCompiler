function main()
{
	let a: [ name: string, age: number ];

	a.name = "Test1";

	print (a.name, " << end (should be value Test1)");

	let b: [ name: string, age: number ] = [ "user", 10.0 ];

	print (b.name, b.age);

	let c: [ user: [ name: string, age: number ], type: number ] = [ [ "user2", 11.0 ], 1.0 ];

	print (c.user.name, c.user.age, c.type);
	print (c.user.name);

	c.user.name = "Test2";

	print (c.user.name, " << end (should be value Test2)");
}