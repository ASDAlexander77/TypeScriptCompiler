#ifndef MLIR_TYPESCRIPT_AST_H_
#define MLIR_TYPESCRIPT_AST_H_

#include "TypeScript/Lexer.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include <vector>

namespace typescript
{

  /// Base class for all expression nodes.
  class ExprAST
  {
  public:
    enum ExprASTKind
    {
      Expr_Num
    };

    ExprAST(ExprASTKind kind, Location location)
        : kind(kind), location(location) {}
    virtual ~ExprAST() = default;

    ExprASTKind getKind() const { return kind; }

    const Location &loc() { return location; }

  private:
    const ExprASTKind kind;
    Location location;
  };

  /// A block-list of expressions.
  using ExprASTList = std::vector<std::unique_ptr<ExprAST>>;

  /// Expression class for numeric literals like "1.0".
  class NumberExprAST : public ExprAST
  {
    double Val;

  public:
    NumberExprAST(Location loc, double val) : ExprAST(Expr_Num, loc), Val(val) {}

    double getValue() { return Val; }

    /// LLVM style RTTI
    static bool classof(const ExprAST *c) { return c->getKind() == Expr_Num; }
  };

  /// This class represents a list of functions to be processed together
  class ModuleAST
  {
    std::vector<std::unique_ptr<ExprAST>> records;

  public:
    ModuleAST(std::vector<std::unique_ptr<ExprAST>> records)
        : records(std::move(records)) {}

    auto begin() -> decltype(records.begin()) { return records.begin(); }
    auto end() -> decltype(records.end()) { return records.end(); }
  };

  void dump(ModuleAST &);

} // namespace typescript

#endif // MLIR_TYPESCRIPT_AST_H_
