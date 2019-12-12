/* Include Guard */
#ifndef INCLUDE_BENZINA_ISO_BMFF_TYPES_H
#define INCLUDE_BENZINA_ISO_BMFF_TYPES_H


/**
 * Includes
 */

#include <stdint.h>


/* Defines */
#define BENZ_MAKE_BOX_TYPE_BYTES(b0, b1, b2, b3)   \
    ((uint32_t)(((b0)   & 0xFFUL) << 24 |          \
                ((b1)   & 0xFFUL) << 16 |          \
                ((b2)   & 0xFFUL) <<  8 |          \
                ((b3)   & 0xFFUL) <<  0))
#define BENZ_MAKE_BOX_TYPE(s) BENZ_MAKE_BOX_TYPE_BYTES((s)[0], (s)[1], (s)[2], (s)[3])


/* Data Structures Forward Declarations and Typedefs */
typedef enum BENZ_ISO_BOX_TYPE         BENZ_ISO_BOX_TYPE;
typedef void                           BENZ_ISO_BOX;
typedef void                           BENZ_ISO_FULLBOX;



/**
 * @brief ISO BMFF read support
 */

enum BENZ_ISO_BOX_TYPE{
    #define MKENUM(type,a,b,c,d)  BENZ_ISO_BOX_TYPE_##type = BENZ_MAKE_BOX_TYPE_BYTES(a, b, c, d)
    MKENUM(assp, 'a','s','s','p'), MKENUM(iinf, 'i','i','n','f'), MKENUM(rloc, 'r','l','o','c'), MKENUM(tols, 't','o','l','s'),
    MKENUM(auxC, 'a','u','x','C'), MKENUM(iloc, 'i','l','o','c'), MKENUM(saio, 's','a','i','o'), MKENUM(traf, 't','r','a','f'),
    MKENUM(auxi, 'a','u','x','i'), MKENUM(imir, 'i','m','i','r'), MKENUM(saiz, 's','a','i','z'), MKENUM(trak, 't','r','a','k'),
    MKENUM(avc1, 'a','v','c','1'), MKENUM(infe, 'i','n','f','e'), MKENUM(sbgp, 's','b','g','p'), MKENUM(tref, 't','r','e','f'),
    MKENUM(avc2, 'a','v','c','2'), MKENUM(ipco, 'i','p','c','o'), MKENUM(schi, 's','c','h','i'), MKENUM(trep, 't','r','e','p'),
    MKENUM(avc3, 'a','v','c','3'), MKENUM(ipma, 'i','p','m','a'), MKENUM(schm, 's','c','h','m'), MKENUM(trex, 't','r','e','x'),
    MKENUM(avc4, 'a','v','c','4'), MKENUM(ipro, 'i','p','r','o'), MKENUM(sdep, 's','d','e','p'), MKENUM(trgr, 't','r','g','r'),
    MKENUM(avcp, 'a','v','c','p'), MKENUM(iprp, 'i','p','r','p'), MKENUM(sdtp, 's','d','t','p'), MKENUM(trun, 't','r','u','n'),
    MKENUM(avcC, 'a','v','c','C'), MKENUM(iref, 'i','r','e','f'), MKENUM(seib, 's','e','i','b'), MKENUM(tsel, 't','s','e','l'),
    MKENUM(avll, 'a','v','l','l'), MKENUM(irot, 'i','r','o','t'), MKENUM(seii, 's','e','i','i'), MKENUM(udta, 'u','d','t','a'),
    MKENUM(avss, 'a','v','s','s'), MKENUM(ispe, 'i','s','p','e'), MKENUM(segr, 's','e','g','r'), MKENUM(url_, 'u','r','l',' '),
    MKENUM(btrt, 'b','t','r','t'), MKENUM(leva, 'l','e','v','a'), MKENUM(sgpd, 's','g','p','d'), MKENUM(urn_, 'u','r','n',' '),
    MKENUM(bxml, 'b','x','m','l'), MKENUM(lhvC, 'l','h','v','C'), MKENUM(sidx, 's','i','d','x'), MKENUM(uuid, 'u','u','i','d'),
    MKENUM(ccst, 'c','c','s','t'), MKENUM(lsel, 'l','s','e','l'), MKENUM(sinf, 's','i','n','f'), MKENUM(vmhd, 'v','m','h','d'),
    MKENUM(clap, 'c','l','a','p'), MKENUM(m4ds, 'm','4','d','s'), MKENUM(skip, 's','k','i','p'), MKENUM(xml_, 'x','m','l',' '),
    MKENUM(co64, 'c','o','6','4'), MKENUM(md5i, 'm','d','5','i'), MKENUM(smhd, 's','m','h','d'),
    MKENUM(colr, 'c','o','l','r'), MKENUM(mdat, 'm','d','a','t'), MKENUM(ssix, 's','s','i','x'),
    MKENUM(cprt, 'c','p','r','t'), MKENUM(mdhd, 'm','d','h','d'), MKENUM(stbl, 's','t','b','l'),
    MKENUM(cslg, 'c','s','l','g'), MKENUM(mdia, 'm','d','i','a'), MKENUM(stco, 's','t','c','o'),
    MKENUM(ctts, 'c','t','t','s'), MKENUM(meco, 'm','e','c','o'), MKENUM(stdp, 's','t','d','p'),
    MKENUM(dinf, 'd','i','n','f'), MKENUM(mehd, 'm','e','h','d'), MKENUM(sthd, 's','t','h','d'),
    MKENUM(dref, 'd','r','e','f'), MKENUM(mere, 'm','e','r','e'), MKENUM(strd, 's','t','r','d'),
    MKENUM(edts, 'e','d','t','s'), MKENUM(meta, 'm','e','t','a'), MKENUM(stri, 's','t','r','i'),
    MKENUM(elng, 'e','l','n','g'), MKENUM(mfhd, 'm','f','h','d'), MKENUM(strk, 's','t','r','k'),
    MKENUM(elst, 'e','l','s','t'), MKENUM(mfra, 'm','f','r','a'), MKENUM(stsc, 's','t','s','c'),
    MKENUM(fecr, 'f','e','c','r'), MKENUM(mfro, 'm','f','r','o'), MKENUM(stsd, 's','t','s','d'),
    MKENUM(fiin, 'f','i','i','n'), MKENUM(minf, 'm','i','n','f'), MKENUM(stsh, 's','t','s','h'),
    MKENUM(fire, 'f','i','r','e'), MKENUM(moof, 'm','o','o','f'), MKENUM(stss, 's','t','s','s'),
    MKENUM(fpar, 'f','p','a','r'), MKENUM(moov, 'm','o','o','v'), MKENUM(stsz, 's','t','s','z'),
    MKENUM(free, 'f','r','e','e'), MKENUM(mvex, 'm','v','e','x'), MKENUM(stts, 's','t','t','s'),
    MKENUM(frma, 'f','r','m','a'), MKENUM(mvhd, 'm','v','h','d'), MKENUM(styp, 's','t','y','p'),
    MKENUM(ftyp, 'f','t','y','p'), MKENUM(nmhd, 'n','m','h','d'), MKENUM(stz2, 's','t','z','2'),
    MKENUM(gitn, 'g','i','t','n'), MKENUM(oinf, 'o','i','n','f'), MKENUM(subs, 's','u','b','s'),
    MKENUM(grpl, 'g','r','p','l'), MKENUM(padb, 'p','a','d','b'), MKENUM(svc1, 's','v','c','1'),
    MKENUM(hev1, 'h','e','v','1'), MKENUM(paen, 'p','a','e','n'), MKENUM(svc2, 's','v','c','2'),
    MKENUM(hdlr, 'h','d','l','r'), MKENUM(pasp, 'p','a','s','p'), MKENUM(svcC, 's','v','c','C'),
    MKENUM(hmhd, 'h','m','h','d'), MKENUM(pdin, 'p','d','i','n'), MKENUM(tfdt, 't','f','d','t'),
    MKENUM(hvcC, 'h','v','c','C'), MKENUM(pitm, 'p','i','t','m'), MKENUM(tfhd, 't','f','h','d'),
    MKENUM(hvc1, 'h','v','c','1'), MKENUM(pixi, 'p','i','x','i'), MKENUM(tfra, 't','f','r','a'),
    MKENUM(idat, 'i','d','a','t'), MKENUM(prft, 'p','r','f','t'), MKENUM(tkhd, 't','k','h','d'),
    #undef  MKENUM
};

#endif

