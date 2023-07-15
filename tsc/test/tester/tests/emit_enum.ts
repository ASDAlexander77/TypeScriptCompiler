import "./decl_enum";

function main()
{
	const p = pointTest (1.0, 2.0);
    if (p.Valid)
	    print(`Valid`)
    if (p.Invalid)
        print(`Invalid`)

    print("done.");
}