import * as ts from 'typescript';
import * as fs from 'fs';

enum SyntaxKindMapped2 {
    FirstAssignment = "EqualsToken",
    LastAssignment = "CaretEqualsToken",
    FirstCompoundAssignment = "PlusEqualsToken",
    LastCompoundAssignment = "CaretEqualsToken",
    FirstReservedWord = "BreakKeyword",
    LastReservedWord = "WithKeyword",
    FirstKeyword = "BreakKeyword",
    LastKeyword = "OfKeyword",
    FirstFutureReservedWord = "ImplementsKeyword",
    LastFutureReservedWord = "YieldKeyword",
    FirstTypeNode = "TypePredicate",
    LastTypeNode = "ImportType",
    FirstPunctuation = "OpenBraceToken",
    LastPunctuation = "CaretEqualsToken",
    FirstToken = "Unknown",
    LastToken = "OfKeyword",
    FirstTriviaToken = "SingleLineCommentTrivia",
    LastTriviaToken = "ConflictMarkerTrivia",
    FirstLiteralToken = "NumericLiteral",
    LastLiteralToken = "NoSubstitutionTemplateLiteral",
    FirstTemplateToken = "NoSubstitutionTemplateLiteral",
    LastTemplateToken = "TemplateTail",
    FirstBinaryOperator = "LessThanToken",
    LastBinaryOperator = "CaretEqualsToken",
    FirstStatement = "VariableStatement",
    LastStatement = "DebuggerStatement",
    FirstNode = "QualifiedName",
    FirstJSDocNode = "JSDocTypeExpression",
    LastJSDocNode = "JSDocPropertyTag",
    FirstJSDocTagNode = "JSDocTag",
    LastJSDocTagNode = "JSDocPropertyTag",
    FirstContextualKeyword = "AbstractKeyword",
    LastContextualKeyword = "OfKeyword"
};

let data;
try {
    fs.statSync(process.argv[2]);
    data = ts.sys.readFile(process.argv[2]);
}
catch (e) {
    data = process.argv[2];
}

const dataStr = "" + data;
const source = ts.createSourceFile("", dataStr, ts.ScriptTarget.Latest);

let intent = 0;
const visitNode = (child) => {

    let s = "";
    for (let i = 0; i < intent; i++)
    {
        s += "\t";
    }

    const strToken = ts.SyntaxKind[child.kind];
    const str2 = SyntaxKindMapped2[strToken];
    console.log(s, "Node:", str2 || strToken, "@ [", child.pos, "-", child.end, "]");

    intent++;
    ts.forEachChild(child, visitNode, visitArray);
    intent--;
};

const visitArray = (nodeArray) => {
    for (const node of nodeArray)
    {
        visitNode(node);
    }    
};

ts.forEachChild(source, visitNode, visitArray);

console.log("Done.");