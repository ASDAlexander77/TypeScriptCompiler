#ifndef TSCLANGIMPORTER_TYPEMAPPER_H
#define TSCLANGIMPORTER_TYPEMAPPER_H

#include "TsClangImporter/BindingModel.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"

namespace clang
{
class Sema;
}
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringSet.h"

#include <utility>
#include <vector>

namespace tsbindgen
{

// C type -> tslang type, for one translation unit. Widths come from the ASTContext, so they are the
// target's: C `long` is i32 for x86_64-pc-windows-msvc and i64 for x86_64-linux-gnu.
class TypeMapper
{
  public:
    enum class Use
    {
        Parameter,
        Result,
        Field,
        Alias // the right-hand side of a typedef
    };

    explicit TypeMapper(clang::ASTContext &context) : context(context)
    {
    }

    // A pointer parameter or field is nullable (`T | null`); a pointer result isn't.
    TsType map(clang::QualType type, Use use);

    // Why a complete struct cannot be a named tuple ("" if it can). Cached; a struct that refers to
    // itself through a pointer counts as mappable while it is being checked.
    std::string recordSkipReason(const clang::RecordDecl *record);

    // A complete struct's fields, as spellRecord lays them out. A pointer field that leads back to
    // this struct is Opaque: tslang cannot declare a type that refers to itself.
    std::vector<Field> mapFields(const clang::RecordDecl *definition);

    // The struct/enum's own name, or the typedef that names it when it has none ("" for neither). A
    // template's specialization is named by its arguments: `static_vector<IBindingSet *, 8>` is
    // `static_vector_IBindingSet_8`.
    static std::string tagName(const clang::TagDecl *tag);
    // tagName with its namespaces and classes for a C++ struct out of the file scope (`nvrhi::Color`);
    // a specialization's has its arguments (`nvrhi::static_vector<nvrhi::IBindingSet *, 8>`)
    static std::string qualifiedTagName(const clang::TagDecl *tag);
    static std::string recordKey(const clang::RecordDecl *record);
    static std::string enumKey(const clang::EnumDecl *enumDecl);
    static std::string typedefKey(const clang::TypedefNameDecl *typedefDecl);

    // `typedef struct S S;` and `typedef struct { ... } S;` name the struct itself: no alias is emitted.
    static bool namesItsTag(const clang::TypedefNameDecl *typedefDecl);

    // A C enum is a TS enum only when it is int-sized and not given a fixed underlying type other
    // than int; anything else is a type alias plus constants.
    bool isPlainEnum(const clang::EnumDecl *enumDecl);

    // Whether a struct can have a TS declaration of its own: in C, any; in C++, one in a namespace or
    // a class too (declared under its own name), but not a template, nor a class local to a
    // function. A template's specialization (std::string, nvrhi::static_vector<T, N>) or a class
    // inside one is declared once a field holds one by value, whose layout needs it; a pointer to
    // one alone stays Opaque.
    bool isDeclarable(const clang::RecordDecl *record);

    // Why a C++ class can't be a @cpp class ("" if it can): it has to be polymorphic, not a
    // template, on a 64-bit target, with at most one base, itself bindable and not virtual - the
    // layout a @cpp class has (the vtable pointer first, a base object the prefix of a derived one).
    std::string classSkipReason(const clang::CXXRecordDecl *record);

    // "class:nvrhi::IDevice"
    static std::string classKey(const clang::CXXRecordDecl *record);

    // A C++ record returned by value, as a method declared to return a tuple returns it: the
    // record's fields, laid out as a tuple's, a pointer as `Opaque | null`. `nonTrivial` is set
    // when C++ can't pass it in registers (a destructor or a copy constructor: @cppnontrivial).
    TsType mapRecordResult(clang::QualType type, bool &nonTrivial);

    // clang's semantic analysis, to complete a class template specialization only ever named as a
    // result (`RefCountPtr<IBuffer> createBuffer(...)` doesn't instantiate it)
    void setSema(clang::Sema *value)
    {
        sema = value;
    }

    // Every struct and enum a mapped type has named, by key. The AST traversal does not reach a
    // struct first named inside a prototype (`void f(struct S *s);`), so the parser adds these.
    const std::vector<std::pair<std::string, const clang::TagDecl *>> &referencedTags() const
    {
        return referenced;
    }

  private:
    // One field of a struct's tuple: its TS name and type, and where C++ puts it (bits).
    struct Slot
    {
        std::string name;
        TsType type;
        uint64_t offset;
        uint64_t size;
        uint64_t align;
    };

    // The tuple a struct is, in layout order: the fields of its bases (non-virtual; an empty one has
    // none), then its own. An array is a tuple of its elements; an anonymous struct's fields are
    // the struct's; a union is its first member when that has the union's size and alignment, raw
    // words otherwise; a run of bitfields raw words (named after the first); an empty
    // [[no_unique_address]] field nothing. Returns why it isn't a tuple, or "".
    std::string spellRecord(const clang::RecordDecl *definition, std::vector<Field> *fields);
    // `baseEnds`: where each base's storage ends (bits), tail padding included
    std::string collectSlots(const clang::RecordDecl *definition, uint64_t base, std::vector<Slot> &slots,
                             std::vector<uint64_t> &baseEnds);
    Slot unionSlot(std::string name, const clang::RecordDecl *unionRecord, uint64_t offset);
    // `[_0: u64, _1: u64, ...]`: `bytes` as words of the largest size up to `align` that fits
    static TsType rawWords(uint64_t bytes, uint64_t align);
    // a template's specialization, or a class inside one, complete: declarable when a field holds it
    bool isInstantiated(const clang::RecordDecl *record);

    TsType mapType(clang::QualType type, Use use);
    std::string referTo(const clang::TagDecl *tag);
    TsType mapPointer(clang::QualType pointee, Use use);
    TsType mapBuiltin(clang::QualType canonical, Use use);
    std::string signatureComment(const clang::FunctionProtoType *function);
    bool isSystem(const clang::Decl *decl);
    const clang::RecordDecl *recordBehind(clang::QualType type);
    bool leadsToOwner(clang::QualType type);

    clang::ASTContext &context;
    clang::Sema *sema = nullptr;
    llvm::DenseMap<const clang::RecordDecl *, std::string> recordReasons;
    llvm::DenseMap<const clang::CXXRecordDecl *, std::string> classReasons;
    std::vector<std::pair<std::string, const clang::TagDecl *>> referenced;
    llvm::StringSet<> referencedKeys;
    std::vector<const clang::RecordDecl *> owners; // structs whose fields are being mapped
    llvm::SmallPtrSet<const clang::RecordDecl *, 16> heldByValue; // instantiated records a field holds
};

} // namespace tsbindgen

#endif // TSCLANGIMPORTER_TYPEMAPPER_H
