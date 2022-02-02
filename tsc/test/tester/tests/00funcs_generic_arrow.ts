function some<T>(arr: T[], f: (it: T) => boolean)
{
	let r = false;
	for (const v of arr) if (r ||= f(v)) break;
	return r;
}

function main() {
    let str = [1, 2, 3];
    assert(some(str, (x => x == 2)), "sometrue");
    assert(!some(str, (x => x < 0)), "somefalse");
    print("done.");
}

