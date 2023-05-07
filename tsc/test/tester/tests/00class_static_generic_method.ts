class Lib
{
	static max<T>(a: T, b: T): T
	{
		return a > b ? a : b;
	}
}

function main() {
    assert(Lib.max(10, 20) == 20);
    print("done.");
}