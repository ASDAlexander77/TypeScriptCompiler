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

    print("done.");
}
