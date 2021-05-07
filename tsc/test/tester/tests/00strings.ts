function display(id:number, name:string)
{
    print("Id = " + id + ", Name = " + name);
}

function main() {                                                 
	display(1, "asd");

	print (("asd" + 1) == ("asd" + "1"));
	print (("asd" + 1) > ("Asd1"));
	print (("asd" + 1) <= ("Asd1"));
	print (("Asd" + 1) < ("asd1"));
	print (("Asd" + 1) >= ("asd1"));
}