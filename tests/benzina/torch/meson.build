pytest = find_program('pytest')

test_devsubset = files('test_devsubset.py')

test('Parse devsubset', pytest, suite: 'torch', args: [test_devsubset], timeout: 120, workdir: meson.source_root())
