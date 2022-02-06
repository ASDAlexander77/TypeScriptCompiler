function some<T>(arr: T[], f: (it: T) => boolean)
{
	let r = false;
	for (const v of arr) if (r ||= f(v)) break;
	return r;
}

function map<T, R>(a: T[], f:(i: T) => R)
{
    let r = 0;
	for (const v of a) r += f(v);
    print(r);
    assert(r == 9);
}

function main() {
    let str = [1, 2, 3];
    assert(some(str, (x => x == 2)), "sometrue");
    assert(!some(str, (x => x < 0)), "somefalse");

	map(str, i => i + 1);

    print("done.");
}

