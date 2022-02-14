import * as ts from 'typescript';

const data = "\
enum Foo {\
    A = 1 << 0,\
    B = 1 << 1,\
}\
";

const dataStr = ("" + data).replace(/\r\n/g, "\n");
const source = ts.createSourceFile("file", dataStr, ts.ScriptTarget.Latest);

console.log("Done.");