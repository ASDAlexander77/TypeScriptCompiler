export class Account
{
	public static data = 10;

	constructor(public n = 0)
	{
		print("account - ctor: ", n);
		this.data = n;
	}
}
