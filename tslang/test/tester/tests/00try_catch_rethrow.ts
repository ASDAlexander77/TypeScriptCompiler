function main() {

    let i = 0;

    try {
        try {
            throw 1;
        }
        catch
        {
            i++;
            print("asd1");
            throw 2;
        }
    }
    catch
    {
        i++;
        print("asd3");
    }

    assert(i == 2);

    print("done.");
}