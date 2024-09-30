function getArray() : Array<string>
{
	return null;
}

function main()
{
	const m = getArray();
	m! && m[0];
	m?.[0]! && m[0];

	assert(!m);
	assert(m === null);

    let arr2 : number[] | undefined;

    let r = arr2?.[10];

    assert (typeof r == "undefined");

    print("done.");
}
