PYTHON=python
SRCFILE=/data/lisa/data/ilsvrc2012.hdf5
TMPDIR=/Tmp/bilaniuo/converter
DSTDIR=/data/milatmp1/bilaniuo/Data

. activate znymky

${PYTHON} ilsvrc2012.py -s ${SRCFILE} -t ${TMPDIR} -d ${DSTDIR} --slice       0   29815 2>/dev/null &
${PYTHON} ilsvrc2012.py -s ${SRCFILE} -t ${TMPDIR} -d ${DSTDIR} --slice   29815   59631 2>/dev/null &
${PYTHON} ilsvrc2012.py -s ${SRCFILE} -t ${TMPDIR} -d ${DSTDIR} --slice   59631   89447 2>/dev/null &
${PYTHON} ilsvrc2012.py -s ${SRCFILE} -t ${TMPDIR} -d ${DSTDIR} --slice   89447  119263 2>/dev/null &
${PYTHON} ilsvrc2012.py -s ${SRCFILE} -t ${TMPDIR} -d ${DSTDIR} --slice  119263  149079 2>/dev/null &
${PYTHON} ilsvrc2012.py -s ${SRCFILE} -t ${TMPDIR} -d ${DSTDIR} --slice  149079  178895 2>/dev/null &
${PYTHON} ilsvrc2012.py -s ${SRCFILE} -t ${TMPDIR} -d ${DSTDIR} --slice  178895  208711 2>/dev/null &
${PYTHON} ilsvrc2012.py -s ${SRCFILE} -t ${TMPDIR} -d ${DSTDIR} --slice  208711  238527 2>/dev/null &
${PYTHON} ilsvrc2012.py -s ${SRCFILE} -t ${TMPDIR} -d ${DSTDIR} --slice  238527  268343 2>/dev/null &
${PYTHON} ilsvrc2012.py -s ${SRCFILE} -t ${TMPDIR} -d ${DSTDIR} --slice  268343  298159 2>/dev/null &
${PYTHON} ilsvrc2012.py -s ${SRCFILE} -t ${TMPDIR} -d ${DSTDIR} --slice  298159  327975 2>/dev/null &
${PYTHON} ilsvrc2012.py -s ${SRCFILE} -t ${TMPDIR} -d ${DSTDIR} --slice  327975  357791 2>/dev/null &
wait
${PYTHON} ilsvrc2012.py -s ${SRCFILE} -t ${TMPDIR} -d ${DSTDIR} --slice  357791  387607 2>/dev/null &
${PYTHON} ilsvrc2012.py -s ${SRCFILE} -t ${TMPDIR} -d ${DSTDIR} --slice  387607  417423 2>/dev/null &
${PYTHON} ilsvrc2012.py -s ${SRCFILE} -t ${TMPDIR} -d ${DSTDIR} --slice  417423  447239 2>/dev/null &
${PYTHON} ilsvrc2012.py -s ${SRCFILE} -t ${TMPDIR} -d ${DSTDIR} --slice  447239  477055 2>/dev/null &
${PYTHON} ilsvrc2012.py -s ${SRCFILE} -t ${TMPDIR} -d ${DSTDIR} --slice  477055  506871 2>/dev/null &
${PYTHON} ilsvrc2012.py -s ${SRCFILE} -t ${TMPDIR} -d ${DSTDIR} --slice  506871  536687 2>/dev/null &
${PYTHON} ilsvrc2012.py -s ${SRCFILE} -t ${TMPDIR} -d ${DSTDIR} --slice  536687  566503 2>/dev/null &
${PYTHON} ilsvrc2012.py -s ${SRCFILE} -t ${TMPDIR} -d ${DSTDIR} --slice  566503  596319 2>/dev/null &
${PYTHON} ilsvrc2012.py -s ${SRCFILE} -t ${TMPDIR} -d ${DSTDIR} --slice  596319  626135 2>/dev/null &
${PYTHON} ilsvrc2012.py -s ${SRCFILE} -t ${TMPDIR} -d ${DSTDIR} --slice  626135  655951 2>/dev/null &
${PYTHON} ilsvrc2012.py -s ${SRCFILE} -t ${TMPDIR} -d ${DSTDIR} --slice  655951  685767 2>/dev/null &
${PYTHON} ilsvrc2012.py -s ${SRCFILE} -t ${TMPDIR} -d ${DSTDIR} --slice  685767  715583 2>/dev/null &
wait
${PYTHON} ilsvrc2012.py -s ${SRCFILE} -t ${TMPDIR} -d ${DSTDIR} --slice  715583  745399 2>/dev/null &
${PYTHON} ilsvrc2012.py -s ${SRCFILE} -t ${TMPDIR} -d ${DSTDIR} --slice  745399  775215 2>/dev/null &
${PYTHON} ilsvrc2012.py -s ${SRCFILE} -t ${TMPDIR} -d ${DSTDIR} --slice  775215  805031 2>/dev/null &
${PYTHON} ilsvrc2012.py -s ${SRCFILE} -t ${TMPDIR} -d ${DSTDIR} --slice  805031  834847 2>/dev/null &
${PYTHON} ilsvrc2012.py -s ${SRCFILE} -t ${TMPDIR} -d ${DSTDIR} --slice  834847  864663 2>/dev/null &
${PYTHON} ilsvrc2012.py -s ${SRCFILE} -t ${TMPDIR} -d ${DSTDIR} --slice  864663  894479 2>/dev/null &
${PYTHON} ilsvrc2012.py -s ${SRCFILE} -t ${TMPDIR} -d ${DSTDIR} --slice  894479  924295 2>/dev/null &
${PYTHON} ilsvrc2012.py -s ${SRCFILE} -t ${TMPDIR} -d ${DSTDIR} --slice  924295  954111 2>/dev/null &
${PYTHON} ilsvrc2012.py -s ${SRCFILE} -t ${TMPDIR} -d ${DSTDIR} --slice  954111  983927 2>/dev/null &
${PYTHON} ilsvrc2012.py -s ${SRCFILE} -t ${TMPDIR} -d ${DSTDIR} --slice  983927 1013743 2>/dev/null &
${PYTHON} ilsvrc2012.py -s ${SRCFILE} -t ${TMPDIR} -d ${DSTDIR} --slice 1013743 1043559 2>/dev/null &
${PYTHON} ilsvrc2012.py -s ${SRCFILE} -t ${TMPDIR} -d ${DSTDIR} --slice 1043559 1073375 2>/dev/null &
wait
${PYTHON} ilsvrc2012.py -s ${SRCFILE} -t ${TMPDIR} -d ${DSTDIR} --slice 1073375 1103191 2>/dev/null &
${PYTHON} ilsvrc2012.py -s ${SRCFILE} -t ${TMPDIR} -d ${DSTDIR} --slice 1103191 1133007 2>/dev/null &
${PYTHON} ilsvrc2012.py -s ${SRCFILE} -t ${TMPDIR} -d ${DSTDIR} --slice 1133007 1162823 2>/dev/null &
${PYTHON} ilsvrc2012.py -s ${SRCFILE} -t ${TMPDIR} -d ${DSTDIR} --slice 1162823 1192639 2>/dev/null &
${PYTHON} ilsvrc2012.py -s ${SRCFILE} -t ${TMPDIR} -d ${DSTDIR} --slice 1192639 1222455 2>/dev/null &
${PYTHON} ilsvrc2012.py -s ${SRCFILE} -t ${TMPDIR} -d ${DSTDIR} --slice 1222455 1252271 2>/dev/null &
${PYTHON} ilsvrc2012.py -s ${SRCFILE} -t ${TMPDIR} -d ${DSTDIR} --slice 1252271 1282087 2>/dev/null &
${PYTHON} ilsvrc2012.py -s ${SRCFILE} -t ${TMPDIR} -d ${DSTDIR} --slice 1282087 1311903 2>/dev/null &
${PYTHON} ilsvrc2012.py -s ${SRCFILE} -t ${TMPDIR} -d ${DSTDIR} --slice 1311903 1341719 2>/dev/null &
${PYTHON} ilsvrc2012.py -s ${SRCFILE} -t ${TMPDIR} -d ${DSTDIR} --slice 1341719 1371535 2>/dev/null &
${PYTHON} ilsvrc2012.py -s ${SRCFILE} -t ${TMPDIR} -d ${DSTDIR} --slice 1371535 1401351 2>/dev/null &
${PYTHON} ilsvrc2012.py -s ${SRCFILE} -t ${TMPDIR} -d ${DSTDIR} --slice 1401351 1431167 2>/dev/null &
wait
