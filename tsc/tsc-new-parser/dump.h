#ifndef DUMP_H
#define DUMP_H

#include <functional>
#include <regex>
#include <string>
#include <iostream>

#include "core.h"
#include "enums.h"
#include "node_test.h"
#include "parser.h"
#include "scanner.h"
#include "scanner_enums.h"
#include "types.h"

namespace ts
{

template <typename OUT> class Printer
{
    OUT &out;
public:    
    Printer(OUT &out) : out(out) {}

    void printLogic(ts::Node node)
    {
        switch (node->_kind)
        {
        case SyntaxKind::Identifier:
            out << node.as<Identifier>()->escapedText;
            break;
        case SyntaxKind::FunctionDeclaration:
            out << "function ";
            break;
        default:
            out << "[MISSING " << Scanner::tokenToText[node->_kind].c_str() << "]";
            break;
        }
    }

    void printNode(ts::Node node)
    {
        ts::FuncT<> visitNode;
        ts::ArrayFuncT<> visitArray;

        auto intent = 0;

        visitNode = [&](ts::Node child) -> ts::Node {
            for (auto i = 0; i < intent; i++)
            {
                std::cout << "\t";
            }

            // print
            printLogic(child);
            ts::forEachChild(child, visitNode, visitArray);

            return undefined;
        };

        visitArray = [&](ts::NodeArray<ts::Node> array) -> ts::Node {
            for (auto node : array)
            {
                visitNode(node);
            }

            return undefined;
        };

        auto result = ts::forEachChild(node, visitNode, visitArray);
    }
};

void printNode(ts::Node node)
{
    Printer<std::wostream> printer(std::wcout);
    printer.printNode(node);
}

} // namespace ts

#endif // DUMP_H