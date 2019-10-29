#
# Linux-specific code.
#

if cc.get_id() == 'pgi' and cc.version().version_compare('>=19.4')
  error('The PGI 19.4 C compiler is known not to support linker scripts well.')
endif

#
# Add implicit linker script.
# 
# Implicit linker scripts are passed through to GNU ld, and can modify the
# visibility of symbols.
#
# On Linux, the only compiler this is known to fail for is PGI 19.4 (though
# 17.3 works), because PGI 19.4 *always* uses a linker script with a VERSION
# section. This is very unfortunate because only one VERSION section can exist.
#
libbenzina_dict += {
    'objs': libbenzina_dict['objs'] + [files('libbenzina.script')],
}
