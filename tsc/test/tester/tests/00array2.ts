function main()
{
	const ac = [1, 2, 3];
	let a = ac;
    a[1] = 9;

	const ac2 = [1.0, 2.0, 3.0];
	let a2 = ac2;
    a2[1] = 9.0;

    const ac3 = ["item 1", "item 2", "item 3"];
	let a3 = ac3;
    a3[1] = "save";

    print("done.");
}
 