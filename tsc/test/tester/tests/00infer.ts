type Flatten<Type> = Type extends Array<infer Item> ? Item : Type;

type GetReturnType<Type> = Type extends (...args: never[]) => infer Return
  ? Return
  : never;
 
type Num = GetReturnType<() => number>;

type ToArrayNonDist<Type> = [Type] extends [unknown] ? Type[] : never;

function main()
{
	let num1: Flatten<number[]> = 20;
	let num2 = 30;

	print(num1!, num2);

	let arr : ToArrayNonDist<string> = ["1", "2"];

    print("done.");
}