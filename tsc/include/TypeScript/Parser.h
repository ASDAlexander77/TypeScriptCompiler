#ifndef MLIR_TYPESCRIPT_PARSER_H
#define MLIR_TYPESCRIPT_PARSER_H

#include "TypeScript/AST.h"
#include "TypeScript/Lexer.h"

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <utility>
#include <vector>

namespace typescript
{

  /// This is a simple recursive parser for the Toy language. It produces a well
  /// formed AST from a stream of Token supplied by the Lexer. No semantic checks
  /// or symbol resolution is performed. For example, variables are referenced by
  /// string and the code could reference an undeclared variable and the parsing
  /// succeeds.
  class Parser
  {
  public:
    /// Create a Parser for the supplied lexer.
    Parser(Lexer &lexer) : lexer(lexer) {}

    /// Parse a full Module. A module is a list of function definitions.
    std::unique_ptr<ModuleAST> parseModule()
    {
      lexer.getNextToken(); // prime the lexer

      // Parse functions and structs one at a time and accumulate in this vector.
      std::vector<std::unique_ptr<ExprAST>> records;
      while (true)
      {
        std::unique_ptr<ExprAST> record;
        switch (lexer.getCurToken())
        {
        case tok_eof:
          break;
        case tok_number:
          record = parseNumberExpr();
          break;
        default:
          return parseError<ModuleAST>("'def' or 'struct'", "when parsing top level module records");
        }
        if (!record)
          break;
        records.push_back(std::move(record));
      }

      // If we didn't reach EOF, there was an error during parsing
      if (lexer.getCurToken() != tok_eof)
        return parseError<ModuleAST>("nothing", "at end of module");

      return std::make_unique<ModuleAST>(std::move(records));
    }

  private:
    Lexer &lexer;

    /// Parse a literal number.
    /// numberexpr ::= number
    std::unique_ptr<ExprAST> parseNumberExpr()
    {
      auto loc = lexer.getLastLocation();
      auto result = std::make_unique<NumberExprAST>(std::move(loc), lexer.getValue());
      lexer.consume(tok_number);
      return std::move(result);
    }

    /// Helper function to signal errors while parsing, it takes an argument
    /// indicating the expected token and another argument giving more context.
    /// Location is retrieved from the lexer to enrich the error message.
    template <typename R, typename T, typename U = const char *>
    std::unique_ptr<R> parseError(T &&expected, U &&context = "")
    {
      auto curToken = lexer.getCurToken();
      llvm::errs() << "Parse error (" << lexer.getLastLocation().line << ", "
                   << lexer.getLastLocation().col << "): expected '" << expected
                   << "' " << context << " but has Token " << curToken;
      if (isprint(curToken))
        llvm::errs() << " '" << (char)curToken << "'";
      llvm::errs() << "\n";
      return nullptr;
    }
  };

} // namespace typescript

#endif // MLIR_TYPESCRIPT_PARSER_H
