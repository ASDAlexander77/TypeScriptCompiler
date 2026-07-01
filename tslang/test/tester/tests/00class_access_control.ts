interface IFace1 
{
    v_public: int;
    v_public_by_default: int;
    v_protected: int;
    get Value(): int;
}

interface IFace2
{
    v_public: int;
    v_public_by_default: int;
    //v_protected: int;
    get Value(): int;
}

class Cls1 implements IFace1
{
    #v_privatefield = 20;
    private v_private = 25;
    protected v_protected = 30;
    public v_public = 40;
    v_public_by_default = 50;

    //static #stat = 123;

    protected print() {
        this.Value = 12;

        this[11] = "asd";

        print(
            this.v_public_by_default, 
            this.v_public, 
            this.v_protected, 
            this.v_private, 
            this.#v_privatefield,
            "...print in Cls1");
    }

    get Value() 
    {
        return this.#v_privatefield;
    }

    protected set Value(val: int)
    {
        this.#v_privatefield;
    }

    [index: number]: string;

    public get(index: number) {
        return "Hello";
    }

    protected set(index: number, val: string) {
        // set value
        print("set value: ", val, " at ", index);
    }
}

class Cls2 extends Cls1 {

    #print() {
        super.print();
    }

    print() {
        this.#print();

        this[444] = "444 - cool";

        print(
            this.v_public_by_default, 
            this.v_public, 
            this.v_protected, 
            //this.v_private, 
            //this.#v_privatefield,
            "...print in Cls2");
    }
}

function main() {
    const cls1 = new Cls1();

    print("Cls1 ...");
    print(cls1.v_public_by_default);
    print(cls1.v_public);
    //print(cls1.v_protected);
    //print(cls1.v_private);
    //print(cls1.#v_privatefield);

    //cls1.print();

    print(cls1.Value);
    //cls1.Value = 12;

    print(cls1[10]);
    //cls1[11] = "asd";

    print("Cls2 ...");
    const cls2 = new Cls2();
    cls2.print();

    //print(Cls1.#stat);

    const iface1: IFace1 = cls1;
    print ("iface1", iface1.v_public, iface1.v_protected, iface1.Value);

    const iface2: IFace2 = cls1;
    print ("iface2", iface2.v_public/*, iface2.v_protected*/, iface2.Value);
   
    print ("done.");
}
  