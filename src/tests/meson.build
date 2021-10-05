#
# Tests
#
# Tests are executables that output the TAP protocol.
#


### C test code compilation
test_lowlevel = executable('test_lowlevel', files('test_lowlevel.c'),
                           dependencies  : [libbenzina])
test_itu      = executable('test_itu',      files('test_itu.c'),
                           dependencies  : [libbenzina])


### Test code run
test('low-level', test_lowlevel, suite: 'low-level', protocol: 'tap')
test('itu',       test_itu,      suite: 'itu',       protocol: 'tap')
test('siphash',   benzina_benz,  suite: 'lua',       protocol: 'tap',
     args: ['lua', files('test_siphash.lua')])
test('bcachefs',  benzina_benz,  suite: 'lua',       protocol: 'tap',
     args: ['lua', files('test_bcachefs.lua')])
