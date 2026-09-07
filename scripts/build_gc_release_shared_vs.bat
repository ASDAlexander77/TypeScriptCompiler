@rem Boehm built as a DLL, installed beside the static one.
@rem
@rem A program that loads a tslang shared library ends up with TWO collectors when both the
@rem executable and the library link gc.lib statically: each has its own heap and its own idea
@rem of what the roots are. Objects the library allocates are then invisible to the executable's
@rem roots, so the collector frees strings the executable is still holding. See item 5ao in
@rem tslang/docs/reference-counting-evaluation.md.
@rem
@rem Static linking stays the default and is correct on its own - one binary, one collector.
@rem This build exists for the shared case, where one collector has to be shared too.
pushd
mkdir __build\gcdll\msbuild\x64\release
cd __build\gcdll\msbuild\x64\release
cmake ../../../../../3rdParty/gc-8.2.12 -G "Visual Studio 18 2026" -A x64 %EXTRA_PARAM% -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -Wno-dev -DCMAKE_INSTALL_PREFIX=../../../../../3rdParty/gcdll/x64/release -Denable_threads=ON -Denable_cplusplus=OFF -DCMAKE_POLICY_DEFAULT_CMP0091=NEW -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded
cmake --build . --config Release -j 8
cmake --install . --config Release
popd
