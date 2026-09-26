#include "TsClangImporter/BindingPrinter.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/GlobPattern.h"

#include <cctype>

namespace tsbindgen
{

namespace
{

// The selected declarations: every seed, then whatever a non-skipped selected declaration uses.
std::vector<bool> selectDecls(const HeaderModel &model, const std::vector<llvm::GlobPattern> &globs)
{
    llvm::StringMap<size_t> byKey;
    for (size_t index = 0; index < model.decls.size(); ++index)
    {
        byKey[model.decls[index].key] = index;
    }

    std::vector<bool> selected(model.decls.size(), false);
    std::vector<size_t> work;
    auto mark = [&](size_t index) {
        if (!selected[index])
        {
            selected[index] = true;
            work.push_back(index);
        }
    };

    for (size_t index = 0; index < model.decls.size(); ++index)
    {
        auto &decl = model.decls[index];
        auto seed = globs.empty() ? decl.inMainFile
                                  : llvm::any_of(globs, [&](const llvm::GlobPattern &glob) { return glob.match(decl.name); });
        if (seed)
        {
            mark(index);
        }
    }

    while (!work.empty())
    {
        auto &decl = model.decls[work.back()];
        work.pop_back();
        if (!decl.skipReason.empty())
        {
            continue; // printed as Opaque or not at all: it uses nothing
        }

        std::vector<std::string> keys;
        for (auto &field : decl.fields)
        {
            collectKeys(field.type, keys);
        }

        collectKeys(decl.type, keys);
        for (auto &key : keys)
        {
            auto found = byKey.find(key);
            if (found != byKey.end())
            {
                mark(found->second);
            }
        }
    }

    return selected;
}

int group(DeclKind kind)
{
    switch (kind)
    {
    case DeclKind::Macro:
        return 0;
    case DeclKind::Function:
        return 2;
    default:
        return 1; // types, in declaration order, before any function uses them
    }
}

} // namespace

llvm::Expected<PrintResult> printBindings(const HeaderModel &model, const PrintOptions &options)
{
    std::vector<llvm::GlobPattern> globs;
    for (auto &filter : options.filters)
    {
        auto glob = llvm::GlobPattern::create(filter);
        if (!glob)
        {
            return glob.takeError();
        }

        globs.push_back(std::move(*glob));
    }

    auto selected = selectDecls(model, globs);
    PrintResult result;
    if (llvm::none_of(selected, [](bool isSelected) { return isSelected; }))
    {
        result.warnings.push_back(options.filters.empty()
                                      ? "nothing to emit: the input file itself declares nothing; pick "
                                        "declarations from its includes with --filter"
                                      : "nothing to emit: no declaration matches --filter");
    }

    // TS names: the C name, minus --strip-prefix unless that leaves nothing, a leading digit, or a
    // name another declaration already has
    llvm::StringMap<std::string> tsNames;
    llvm::StringMap<std::string> taken;
    for (size_t index = 0; index < model.decls.size(); ++index)
    {
        if (!selected[index])
        {
            continue;
        }

        auto &decl = model.decls[index];
        auto name = decl.name;
        llvm::StringRef stripped(name);
        if (!options.stripPrefix.empty() && stripped.consume_front(options.stripPrefix) && !stripped.empty() &&
            !std::isdigit(static_cast<unsigned char>(stripped.front())))
        {
            auto candidate = escapeReserved(stripped.str());
            if (taken.count(candidate))
            {
                result.warnings.push_back("kept '" + name + "' unstripped: '" + candidate + "' is taken by '" +
                                          taken[candidate] + "'");
            }
            else
            {
                name = candidate;
            }
        }

        // a legal C name TS reserves (`int new(void)`), and a type named like one of tslang's own,
        // which tslang would ignore in favour of its own type (`struct string`), get a `_`
        name = escapeReserved(name);
        if (decl.kind != DeclKind::Function && decl.kind != DeclKind::Macro && isTsBuiltinTypeName(name))
        {
            name += "_";
        }

        taken[name] = decl.name;
        tsNames[decl.key] = name;
    }

    auto nameOf = [&](const std::string &key) {
        auto found = tsNames.find(key);
        return found != tsNames.end() ? found->second : nameFromKey(key);
    };

    auto inNamespace = !options.namespaceName.empty();
    auto indent = std::string(inNamespace ? "    " : "");
    auto exported = std::string(inNamespace ? "export " : "");
    std::string body;
    auto line = [&](const std::string &text) { body += indent + text + "\n"; };
    auto hasFunctionPointer = false;
    auto withComment = [&](const Field &field) {
        auto text = field.name + ": " + renderType(field.type, nameOf);
        if (!field.type.comment.empty())
        {
            hasFunctionPointer |= field.type.functionPointer;
            text += " /* " + field.type.comment + " */";
        }

        return text;
    };

    for (int pass = 0; pass < 3; ++pass)
    {
        for (size_t index = 0; index < model.decls.size(); ++index)
        {
            auto &decl = model.decls[index];
            if (!selected[index] || group(decl.kind) != pass)
            {
                continue;
            }

            auto tsName = nameOf(decl.key);
            if (!decl.skipReason.empty())
            {
                line("// skipped: " + decl.name + " \xE2\x80\x94 " + decl.skipReason);
                result.warnings.push_back("skipped " + decl.name + ": " + decl.skipReason);
                // a skipped struct keeps its users: a pointer to it is Opaque
                if (decl.kind == DeclKind::Struct)
                {
                    line(exported + "type " + tsName + " = Opaque;");
                }

                continue;
            }

            switch (decl.kind)
            {
            case DeclKind::Macro:
                line(exported + "const " + tsName + " = " + decl.value + ";");
                break;
            case DeclKind::OpaqueStruct:
                line(exported + "type " + tsName + " = Opaque;");
                break;
            case DeclKind::Struct: {
                std::string fields;
                for (auto &field : decl.fields)
                {
                    fields += (fields.empty() ? "" : ", ") + withComment(field);
                }

                line(exported + "type " + tsName + " = [" + fields + "];");
                break;
            }
            case DeclKind::Enum: {
                std::string members;
                for (auto &enumerator : decl.enumerators)
                {
                    members += (members.empty() ? "" : ", ") + enumerator.name + " = " + enumerator.value;
                }

                line(exported + "enum " + tsName + " { " + members + " }");
                break;
            }
            case DeclKind::FixedEnum:
                line(exported + "type " + tsName + " = " + renderType(decl.type, nameOf) + ";");
                for (auto &enumerator : decl.enumerators)
                {
                    auto constName = decl.scoped ? tsName + "_" + enumerator.name : enumerator.name;
                    line(exported + "const " + constName + " = " + enumerator.value + ";");
                }

                break;
            case DeclKind::Typedef: {
                auto text = exported + "type " + tsName + " = " + renderType(decl.type, nameOf) + ";";
                if (!decl.type.comment.empty())
                {
                    hasFunctionPointer |= decl.type.functionPointer;
                    text += " // " + decl.type.comment;
                }

                line(text);
                break;
            }
            case DeclKind::Function: {
                std::string parameters;
                for (auto &parameter : decl.fields)
                {
                    parameters += (parameters.empty() ? "" : ", ") + withComment(parameter);
                }

                std::string decorators = decl.varargs ? "@varargs " : "";
                // a namespace qualifies the symbol, and a TS name can differ from the C one
                // (stripped, escaped) or the C name from the symbol (an asm label)
                auto symbol = decl.symbol.empty() ? decl.name : decl.symbol;
                if (inNamespace || tsName != symbol)
                {
                    decorators += "@linkname(\"" + symbol + "\") ";
                }

                line(decorators + exported + "declare function " + tsName + "(" + parameters +
                     "): " + renderType(decl.type, nameOf) + ";");
                break;
            }
            }
        }
    }

    for (auto &headerLine : options.headerLines)
    {
        result.text += "// " + headerLine + "\n";
    }

    if (hasFunctionPointer)
    {
        result.text += "// A function pointer is Opaque: pass a function without captures, as `fn as Opaque`.\n";
    }

    if (!result.text.empty())
    {
        result.text += "\n";
    }

    if (inNamespace)
    {
        result.text += "namespace " + options.namespaceName + " {\n" + body + "}\n";
    }
    else
    {
        result.text += body;
    }

    return result;
}

} // namespace tsbindgen
