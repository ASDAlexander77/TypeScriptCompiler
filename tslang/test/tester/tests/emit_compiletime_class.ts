@dllimport
declare class Account
{
    	public n?: TypeOf<1>;

	constructor(n?: TypeOf<1>);
}

function main()
{
	const a1 = new Account();
	print(a1.n);

	const a2 = new Account(2);
	print(a2.n);

	print("done.");
}