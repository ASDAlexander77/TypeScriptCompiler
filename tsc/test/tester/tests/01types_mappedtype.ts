type FeatureFlags = {
  darkMode: string;
  newUserProfile: string;
};

type OptionsFlags<Type> = {
  [Property in keyof Type]: boolean;
};


function f<A>(a: A, opt: OptionsFlags<A>)
{
	print(a.darkMode, opt.darkMode);
}


function main()
{
	let a: FeatureFlags = { darkMode: "asd1", newUserProfile: "asd2" };
	f(a, { darkMode: true, newUserProfile: false });

    print("done.");
}