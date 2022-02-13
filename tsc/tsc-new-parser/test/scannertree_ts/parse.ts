import * as ts from 'typescript';

const data = "function main() { for (let i = 0; i < 10; i++); }";

const dataStr = ("" + data).replace(/\r\n/g, "\n");
const source = ts.createSourceFile("file", dataStr, ts.ScriptTarget.Latest);

console.log("Done.");