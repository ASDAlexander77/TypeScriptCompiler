enum EnumInt {
    A = 1,
    B = 2
}

function enumInt()
{
    assert(EnumInt.A == 1);
    assert(EnumInt.B == 2);
}

enum EnumFloat {
    A = 1.0,
    B = 2.0
}

function enumFloat()
{
    assert(EnumFloat.A == 1.0);
    assert(EnumFloat.B == 2.0);
}

enum EnumBool {
    A = true,
    B = false
}

function enumBool()
{
    assert(EnumBool.A == true);
    assert(EnumBool.B == false);
}

enum EnumString {
    A = "str1",
    B = "str2"
}

function enumString()
{
    assert(EnumString.A == "str1");
    assert(EnumString.B == "str2");
}

enum EnumMix {
    A = 1,
    B = "str2"
}

function enumMix()
{
    assert(EnumMix.A == 1);
    assert(EnumMix.B == "str2");
}

function main() 
{
    enumInt();
    enumFloat();
    enumBool();
    enumString();
    enumMix();
    print("done.");
}

