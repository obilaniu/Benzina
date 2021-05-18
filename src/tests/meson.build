#
# Tests
#
# Tests are executables that output the TAP protocol.
#


### C test code compilation
test_itu      = executable('test_itu',      files('test_itu.c'),
                           dependencies  : [libbenzina])
test_lowlevel = executable('test_lowlevel', files('test_lowlevel.c'),
                           dependencies  : [libbenzina])
test_lua      = executable('test_lua',      files('test_lua.c'),
                           dependencies  : [libbenzina])


### Test code run
test('itu',       test_itu,      suite: 'itu',       protocol: 'tap')
test('low-level', test_lowlevel, suite: 'low-level', protocol: 'tap')
test('lua',       test_lua,      suite: 'lua',       protocol: 'tap')
