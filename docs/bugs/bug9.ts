function main() {

    print("start...");

	let i = 2;

	let f = () => { return i; };

	for (let i = 0; i < 1000000; i++)
	{
		let r = f() + i;
		print (`val : ${r}`);
	}

    print("done.");
}
                                                                     