function main()
{
	const ac = [1, 2, 3];
	let a = ac;
	print(ac[0]);
	print(ac[1]);
	print(ac[2]);
	print(a[0]);
	print(a[1]);
	print(a[2]);
    a[1] = 9;

	const ac2 = [1.0, 2.0, 3.0];
	let a2 = ac2;
	print(ac2[0]);
	print(ac2[1]);
	print(ac2[2]);

	print(a2[0]);
	print(a2[1]);
	print(a2[2]);
    a2[1] = 9.0;

    const ac3 = ["item 1", "item 2", "item 3"];
	let a3 = ac3;
	print(ac3[0]);
	print(ac3[1]);
	print(ac3[2]);

	print(a3[0]);
	print(a3[1]);
	print(a3[2]);
    a3[1] = "save";
}
