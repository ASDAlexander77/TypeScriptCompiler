function t_undef(s?: string)
{
	assert((s == undefined) == true, "is not undefined")
	assert((s != undefined) == false, "is undefined")
	assert((s >  undefined) == false, "not >");
	assert((s <  undefined) == false, "not <");
	assert((s >= undefined) == true, "not >=");
	assert((s <= undefined) == true, "not <=");
}

function t_val(s?: string)
{
	assert((s == undefined) == false, "is undefined")
	assert((s != undefined) == true, "is not undefined")
	assert((s >  undefined) == true, ">");
	assert((s <  undefined) == false, "<");
	assert((s >= undefined) == true, ">=");
	assert((s <= undefined) == false, "<=");
}

function f(s?: string)
{
	print(s == undefined, s != undefined, s > undefined, s < undefined, s >= undefined, s <= undefined);
}

function main() {
	f();
	t_undef();
	f("asd");
	t_val("asd");

    print("done.");    
}

