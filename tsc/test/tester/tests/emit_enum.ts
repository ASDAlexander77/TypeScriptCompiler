import "./decl_enum";

function main()
{
	const p = pointTest (1.0, 2.0);
    if (p == PointTest.Valid)
	    print(`Valid`)
    if (p == PointTest.Invalid)
        print(`Invalid`)

    print("done.");
}