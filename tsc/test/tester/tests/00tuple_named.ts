function main()
{
	let a: [ string, number ];
	let b: [ name: string, age: number ] = [ "user", 10.0 ];

	//b.name = "User1";
	
	print (b.name, b.age);

	let c: [ user: [ name: string, age: number ], type: number ] = [ [ "user2", 11.0 ], 1.0 ];

	print (c.user.name, c.user.age, c.type);
}