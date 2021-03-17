import * as ts from 'typescript';
import * as fs from 'fs';

const data = fs.readFileSync(process.argv[2]);

const scanner = ts.createScanner(ts.ScriptTarget.Latest, true, ts.LanguageVariant.Standard, ""+data);

let token = ts.SyntaxKind.Unknown;
while (token != ts.SyntaxKind.EndOfFileToken)
{
	token = scanner.scan();
	console.log(token,"@", scanner.getTokenPos(), scanner.getTokenText());
}