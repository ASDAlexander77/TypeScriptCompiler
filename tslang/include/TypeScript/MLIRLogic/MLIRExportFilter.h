#ifndef TYPESCRIPT_MLIRGENLOGIC_EXPORTFILTER_H
#define TYPESCRIPT_MLIRGENLOGIC_EXPORTFILTER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/GlobPattern.h"

#include <string>

namespace typescript
{
    // A filter is `all`, `none`, a name or glob matched against the short or namespaced
    // name, or any of those prefixed with `!` to exclude.
    inline bool matchesExportFilter(llvm::StringRef pattern, llvm::StringRef name, llvm::StringRef fullName)
    {
        if (pattern == "all")
        {
            return true;
        }

        if (pattern == "none")
        {
            return false;
        }

        auto glob = llvm::GlobPattern::create(pattern);
        if (!glob)
        {
            llvm::consumeError(glob.takeError());
            return pattern == name || pattern == fullName;
        }

        return (!name.empty() && glob->match(name)) || (!fullName.empty() && glob->match(fullName));
    }

    // Exclusions always win; without any including filter the `export` keyword decides.
    inline bool isExportedByFilters(llvm::ArrayRef<std::string> filters, llvm::StringRef name, llvm::StringRef fullName,
                                    bool hasExportKeyword)
    {
        auto hasIncluding = false;
        auto included = false;
        for (auto &filter : filters)
        {
            llvm::StringRef pattern(filter);
            if (pattern.consume_front("!"))
            {
                if (matchesExportFilter(pattern, name, fullName))
                {
                    return false;
                }

                continue;
            }

            hasIncluding = true;
            included |= matchesExportFilter(pattern, name, fullName);
        }

        return hasIncluding ? included : hasExportKeyword;
    }
} // namespace typescript

#endif // TYPESCRIPT_MLIRGENLOGIC_EXPORTFILTER_H
