#include "TypeScript/LibraryDeclarations.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Path.h"

namespace typescript
{

static const llvm::StringRef modulePrefix = "//@tslang-module ";
static const llvm::StringRef declarationsLine = "//@tslang-declarations";
static const llvm::StringRef genericLine = "//@tslang-generic";
static const llvm::StringRef ownFactsLine = "//@tslang-own-facts";

std::string libraryDeclarationsPath(llvm::StringRef binaryPath)
{
    llvm::SmallString<256> path(binaryPath);
    llvm::sys::path::replace_extension(path, ".d.ts");
    return std::string(path);
}

static void appendText(std::string &out, llvm::StringRef text)
{
    out += text;
    if (!text.empty() && !text.ends_with("\n"))
    {
        out += "\n";
    }
}

std::string printLibraryDeclarations(const std::vector<LibraryModuleDeclarations> &modules)
{
    std::string out;
    for (auto &module : modules)
    {
        out += LIBRARY_DECLARATIONS_FILE_MARKER "\n";
        out += modulePrefix;
        out += module.suffix;
        out += " model=";
        out += module.memoryModel;
        if (module.hasGCtors)
        {
            out += " gctors";
        }

        out += "\n";

        if (!module.declarations.empty())
        {
            out += declarationsLine;
            out += "\n";
            appendText(out, module.declarations);
        }

        if (!module.genericDeclarations.empty())
        {
            out += genericLine;
            out += "\n";
            appendText(out, module.genericDeclarations);
        }

        if (!module.ownFacts.empty())
        {
            out += ownFactsLine;
            out += "\n";
            llvm::SmallVector<llvm::StringRef> names;
            llvm::StringRef(module.ownFacts).split(names, '\n', -1, false);
            for (auto name : names)
            {
                name = name.trim();
                if (!name.empty())
                {
                    out += "// ";
                    out += name;
                    out += "\n";
                }
            }
        }
    }

    return out;
}

bool isLibraryDeclarationsText(llvm::StringRef text)
{
    // a UTF-8 BOM, should an editor have added one
    text.consume_front("\xEF\xBB\xBF");
    return text.starts_with(LIBRARY_DECLARATIONS_FILE_MARKER);
}

std::optional<std::vector<LibraryModuleDeclarations>> parseLibraryDeclarations(llvm::StringRef text)
{
    if (!isLibraryDeclarationsText(text))
    {
        return std::nullopt;
    }

    enum class Section
    {
        None,
        Declarations,
        Generic,
        OwnFacts
    };

    std::vector<LibraryModuleDeclarations> modules;
    auto section = Section::None;

    llvm::SmallVector<llvm::StringRef> lines;
    text.split(lines, '\n');
    // the text after the last line's end: no line
    if (!lines.empty() && lines.back().empty())
    {
        lines.pop_back();
    }

    for (auto line : lines)
    {
        line.consume_back("\r");
        line.consume_front("\xEF\xBB\xBF");

        // the first line of every block (blocks of several objects follow each other)
        if (line == LIBRARY_DECLARATIONS_FILE_MARKER)
        {
            section = Section::None;
            continue;
        }

        if (line.starts_with(modulePrefix))
        {
            LibraryModuleDeclarations module;
            llvm::SmallVector<llvm::StringRef> words;
            line.drop_front(modulePrefix.size()).split(words, ' ', -1, false);
            for (auto word : words)
            {
                if (module.suffix.empty())
                {
                    module.suffix = word.str();
                }
                else if (word.consume_front("model="))
                {
                    module.memoryModel = word.str();
                }
                else if (word == "gctors")
                {
                    module.hasGCtors = true;
                }
            }

            modules.push_back(std::move(module));
            section = Section::None;
            continue;
        }

        if (line == declarationsLine)
        {
            section = Section::Declarations;
            continue;
        }

        if (line == genericLine)
        {
            section = Section::Generic;
            continue;
        }

        if (line == ownFactsLine)
        {
            section = Section::OwnFacts;
            continue;
        }

        if (modules.empty())
        {
            continue;
        }

        auto &module = modules.back();
        switch (section)
        {
        case Section::Declarations:
            module.declarations += line;
            module.declarations += "\n";
            break;
        case Section::Generic:
            module.genericDeclarations += line;
            module.genericDeclarations += "\n";
            break;
        case Section::OwnFacts:
            if (line.consume_front("//"))
            {
                line = line.trim();
                if (!line.empty())
                {
                    module.ownFacts += line;
                    module.ownFacts += "\n";
                }
            }
            break;
        case Section::None:
            break;
        }
    }

    return modules;
}

} // namespace typescript
