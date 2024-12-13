class Cls1
{
    test: int;

    protected constructor()
    {
        print("protected ctor");
        this.test = 111;
    }

    public static getCls1()
    {
        return new Cls1();
    }
}

//const cls0 = new Cls1();
//print(cls0.test); // error

const cls1 = Cls1.getCls1();
print(cls1.test);
assert(cls1.test == 111);

print("done.");