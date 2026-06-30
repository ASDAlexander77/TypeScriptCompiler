class Cls1
{
    public v = 10;
}

class t
{
    constructor(public a = new Cls1()) {}
}

function main() {
    const ti = new t();
    assert(ti.a.v == 10);
    print ("done.");
}
  