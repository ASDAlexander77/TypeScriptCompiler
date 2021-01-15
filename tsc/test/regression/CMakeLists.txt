configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(TYPESCRIPT_TEST_DEPENDS
        FileCheck count not
        tsc-opt
        tsc-translate
        )

add_lit_testsuite(check-typescript "Running the typescript regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${TYPESCRIPT_TEST_DEPENDS}
        )
set_target_properties(check-typescript PROPERTIES FOLDER "Tests")

add_lit_testsuites(TYPESCRIPT ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${TYPESCRIPT_TEST_DEPENDS})
