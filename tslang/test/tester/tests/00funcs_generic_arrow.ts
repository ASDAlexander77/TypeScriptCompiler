function some<T>(arr: T[], f: (it: T) => boolean)
{
	for (const v of arr) if (f(v)) return true;
	return false;
}

function map<T, R>(a: T[], f:(i: T) => R)
{
    let r = 0;
	for (const v of a) r += f(v);
    print(r);
    assert(r == 9);
}

function reduce<T, R>(arr: T[], f: (s: R, v: T) => R, init: R)
{
	let r = init;
	for (const v of arr) r = f(r, v);
	return r;
}

function main() {
    let str = [1, 2, 3];
    assert(some(str, (x => x == 2)), "sometrue");
    assert(!some(str, (x => x < 0)), "somefalse");

	map(str, i => i + 1);

	let count = 0;
	map([1, 2, 3], (i) => { count++; return i + 1; });
	assert(count == 3);

    let sum = reduce([1, 2, 3], (s, v) => s + v, 0);
    assert(sum == 6, "red")

    print("done.");
}

