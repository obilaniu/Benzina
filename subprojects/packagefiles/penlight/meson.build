project('Penlight', 'c',
    version:         '1.10.0',
    meson_version:   '>=0.58.1',
    license:         'MIT',
)


penlight_lua = {'pl': files('lua'/'pl'/'init.lua')}
foreach p: ['app',           'array2d',
            'class',         'compat',
            'comprehension', 'config',
            'data',          'Date',
            'dir',           'file',
            'func',          'import_into',
            'input',         'lapp',
            'lexer',         'List',
            'luabalanced',   'Map',
            'MultiMap',      'operator',
            'OrderedMap',    'path',
            'permute',       'pretty',
            'seq',           'Set',
            'sip',           'strict',
            'stringio',      'stringx',
            'tablex',        'template',
            'test',          'text',
            'types',         'url',
            'utils',         'xml']
  penlight_lua += {'pl.'+p: files('lua'/'pl'/p+'.lua')}
endforeach


penlight_dep = declare_dependency(
    link_whole: [
        static_library('penlight-linker', files('linker.noarch.S'), pic: true, name_prefix: ''),
    ],
)
