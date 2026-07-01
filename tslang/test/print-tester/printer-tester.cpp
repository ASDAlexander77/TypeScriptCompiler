#include <cstdio> 

// parser includes
#include "enums.h"
#include "dump.h"
#include "file_helper.h"
#include "node_factory.h"
#include "parser.h"
#include "utilities.h"

using namespace ts;

void getNodes(ts::SourceFile sourceFile, NodeArray<Node> &nodes)
{
    FuncT<> visitNode;
    ArrayFuncT<> visitArray;

    visitNode = [&](Node child) -> Node {

        nodes.push_back(child);

        ts::forEachChild(child, visitNode, visitArray);
        return undefined;
    };

    visitArray = [&](NodeArray<Node> array) -> Node {
        for (auto node : array)
        {
            visitNode(node);
        }

        return undefined;
    };

    forEachChild(sourceFile.as<Node>(), visitNode, visitArray);
}

int main(int argc, char **argv)
{
    if (argc <= 1)
    {
        puts("Usage: print-tester <file path>.ts [compare|declare]?");
        return 0;
    }

    auto declareMode = argc > 1 && std::string(argv[2]) == "declare";
    auto compareMode = argc > 1 && std::string(argv[2]) == "compare";

    std::string filePath(argv[1]);

    puts(argv[1]);

    auto content = readFile(filePath);

    Parser parser;
    auto sourceFile = parser.parseSourceFile(stows(filePath), content, ScriptTarget::Latest);

    stringstream out;
    Printer<stringstream> printer(out);
    printer.setDeclarationMode(declareMode);
    printer.printNode(sourceFile);

    auto newContent = out.str();
    if (!compareMode)
    {
        puts(wstos(newContent).c_str());
    }
    else
    {
        // compare mode
        auto newSourceFile = parser.parseSourceFile(stows(filePath), newContent, ScriptTarget::Latest);

        NodeArray<Node> oldNodes;
        NodeArray<Node> newNodes;

        getNodes(sourceFile, oldNodes);
        getNodes(newSourceFile, newNodes);

        auto count = oldNodes.size() > newNodes.size() ? oldNodes.size() : newNodes.size();

        if (count == 0)
        {
            puts(" : ERROR, count = 0");
            exit(1);
        }

        if (oldNodes.size() != newNodes.size())
        {
            puts(" : not equal count.");
        }

        for (auto index = 0; index < count; index++)
        {
            auto oldNode = oldNodes[index];
            auto newNode = newNodes[index];

            // compare
            if ((SyntaxKind)oldNode != (SyntaxKind)newNode)
            {
                auto posLineChar = parser.getLineAndCharacterOfPosition(sourceFile, oldNode->pos);
                auto endLineChar = parser.getLineAndCharacterOfPosition(sourceFile, oldNode->_end);

                stringstream s;
                s << S("Node: ") << parser.syntaxKindString(oldNode).c_str() << S(" @ [ ") << oldNode->pos << S("(")
                << posLineChar.line + 1 << S(":") << posLineChar.character + 1 << S(") - ") << oldNode->_end << S("(")
                << endLineChar.line + 1 << S(":") << endLineChar.character << S(") ]")
                << std::endl << " old text: " << content.substr(oldNode->pos, oldNode->_end - oldNode->pos)
                << std::endl << " new text: " << newContent.substr(newNode->pos, newNode->_end - newNode->pos);

                puts(wstos(s.str()).c_str());
                puts(" : not equal.");
                exit(1);
            }
        }

        puts(" : equal.");
    }

    return 0;
}
