class ConcreteA {}
class ConcreteB {}

function main()
{
	[ConcreteA, ConcreteB].map(cls => new cls()); // should work
	print("done.");
}