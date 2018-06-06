/* Includes */
#include <stdlib.h>
#include <string.h>
#include "benzinaproto.h"





/* Function Definitions */
BENZINA_HIDDEN int          benzinaBufInit     (BENZINA_BUF*  bbuf){
	bbuf->buf    = NULL;
	bbuf->off    = 0;
	bbuf->len    = 0;
	bbuf->maxLen = 0;
	return 0;
}
BENZINA_HIDDEN int          benzinaBufFini     (BENZINA_BUF*  bbuf){
	if(bbuf){
		free(bbuf->buf);
		memset(bbuf, 0, sizeof(*bbuf));
	}
	return 0;
}

BENZINA_HIDDEN int          benzinaBufEnsure   (BENZINA_BUF*  bbuf, size_t freeSpace){
	size_t newMaxLen = bbuf->off+freeSpace;
	void*  newBuf    = bbuf->buf;
	
	if(newMaxLen > bbuf->maxLen){
		newMaxLen = newMaxLen > 2*bbuf->maxLen ? newMaxLen : 2*bbuf->maxLen;
		newBuf    = realloc(bbuf->buf, newMaxLen);
		if(newBuf){
			bbuf->buf    = newBuf;
			bbuf->maxLen = newMaxLen;
		}
	}
	
	return !newBuf;
}

/* Raw Read/Write calls. */
BENZINA_HIDDEN int          benzinaBufRead     (BENZINA_BUF*  bbuf,
                                                char*         data,
                                                size_t        len){
	if(len > bbuf->len-bbuf->off){
		return 1;
	}else{
		memcpy(data, bbuf->buf+bbuf->off, len);
		bbuf->off += len;
		return 0;
	}
}
BENZINA_HIDDEN int          benzinaBufWrite    (BENZINA_BUF*  bbuf,
                                                const char*   data,
                                                size_t        len){
	int ret;
	
	ret = benzinaBufEnsure(bbuf, len);
	if(ret){
		return ret;
	}else{
		if(bbuf->off+len > bbuf->len){
			bbuf->len = bbuf->off+len;
		}
		memcpy(bbuf->buf+bbuf->off, data, len);
		bbuf->off += len;
		return 0;
	}
}

/* Protobuf */
/* Protobuf Read */
BENZINA_HIDDEN int          benzinaBufReadLDL  (BENZINA_BUF*  bbuf,
                                                const char**  data,
                                                size_t*       len){
	uint64_t v;
	int      ret;
	ret = benzinaBufReadvu64(bbuf, &v);
	*len  = v;
	*data = bbuf->buf;
	bbuf->buf += v;
	return ret;
}
BENZINA_HIDDEN int          benzinaBufReadStr  (BENZINA_BUF*  bbuf,
                                                char**        str,
                                                size_t*       len){
	int   ret;
	char* nulterm;
	
	ret     = benzinaBufReadLDL(bbuf, (const char**)str, len);
	if(ret){
		return ret;
	}
	
	nulterm = malloc(*len);
	if(nulterm){
		memcpy(nulterm, *str, *len);
		*str = nulterm;
		return 0;
	}else{
		*str = NULL;
		return 1;
	}
}
BENZINA_HIDDEN int          benzinaBufReadMsg  (BENZINA_BUF*  bbuf,
                                                BENZINA_BUF*  msg){
	int ret;
	
	ret = benzinaBufEnsure(msg, bbuf->len);
	if(ret){
		return ret;
	}
	
	memcpy(msg->buf+msg->len, bbuf->buf, bbuf->len);
	
	return 0;
}

BENZINA_HIDDEN int          benzinaBufReadvu64 (BENZINA_BUF*  bbuf, uint64_t* v){
	char c;
	int  i=0, ret;
	
	*v = 0;
	do{
		ret = benzinaBufRead(bbuf, &c, 1);
		if(ret){
			return ret;
		}
		*v |= (uint64_t)(c&0x7F) << 7*i++;
	}while(i<10 && (c&0x80));
	
	return 0;
}
BENZINA_HIDDEN int          benzinaBufReadvs64 (BENZINA_BUF*  bbuf, int64_t*  v){
	uint64_t w;
	int ret = benzinaBufReadvu64(bbuf, &w);
	*v = (w >> 1)^-(w&1);
	return ret;
}
BENZINA_HIDDEN int          benzinaBufReadvi64 (BENZINA_BUF*  bbuf, int64_t*  v){
	return benzinaBufReadvu64(bbuf, (uint64_t*)v);
}
BENZINA_HIDDEN int          benzinaBufReadvu32 (BENZINA_BUF*  bbuf, uint32_t* v){
	uint64_t w;
	int ret = benzinaBufReadvu64(bbuf, &w);
	*v = w;
	return ret;
}
BENZINA_HIDDEN int          benzinaBufReadvs32 (BENZINA_BUF*  bbuf, int32_t*  v){
	uint32_t w;
	int ret = benzinaBufReadvu32(bbuf, &w);
	*v = (w >> 1)^-(w&1);
	return ret;
}
BENZINA_HIDDEN int          benzinaBufReadvi32 (BENZINA_BUF*  bbuf, int32_t*  v){
	return benzinaBufReadvu32(bbuf, (uint32_t*)v);
}

BENZINA_HIDDEN int          benzinaBufReadfu64 (BENZINA_BUF*  bbuf, uint64_t* u){
	return benzinaBufRead(bbuf, (char*)u, sizeof(*u));
}
BENZINA_HIDDEN int          benzinaBufReadfs64 (BENZINA_BUF*  bbuf, int64_t*  s){
	return benzinaBufRead(bbuf, (char*)s, sizeof(*s));
}
BENZINA_HIDDEN int          benzinaBufReadff64 (BENZINA_BUF*  bbuf, double*   f){
	return benzinaBufRead(bbuf, (char*)f, sizeof(*f));
}
BENZINA_HIDDEN int          benzinaBufReadfu32 (BENZINA_BUF*  bbuf, uint32_t* u){
	return benzinaBufRead(bbuf, (char*)u, sizeof(*u));
}
BENZINA_HIDDEN int          benzinaBufReadfs32 (BENZINA_BUF*  bbuf, int32_t*  s){
	return benzinaBufRead(bbuf, (char*)s, sizeof(*s));
}
BENZINA_HIDDEN int          benzinaBufReadff32 (BENZINA_BUF*  bbuf, float*    f){
	return benzinaBufRead(bbuf, (char*)f, sizeof(*f));
}

BENZINA_HIDDEN int          benzinaBufReadEnum (BENZINA_BUF*  bbuf, int32_t*  e){
	return benzinaBufReadvi32(bbuf, e);
}
BENZINA_HIDDEN int          benzinaBufReadBool (BENZINA_BUF*  bbuf, int*      b){
	uint32_t v;
	int ret = benzinaBufReadvu32(bbuf, &v);
	*b = !!v;
	return ret;
}

BENZINA_HIDDEN int          benzinaBufReadTagW (BENZINA_BUF*  bbuf,
                                                uint32_t*     tag,
                                                uint32_t*     wire){
	uint32_t v;
	int ret = benzinaBufReadvu32(bbuf, &v);
	*tag  = v >> 3;
	*wire = v &  0x7;
	return ret;
}

/* Protobuf Write */
BENZINA_HIDDEN int          benzinaBufWriteLDL (BENZINA_BUF*  bbuf,
                                                const char*   data,
                                                size_t        len){
	return benzinaBufWritevi64(bbuf, len) ||
	       benzinaBufWrite    (bbuf, data, len);
}
BENZINA_HIDDEN int          benzinaBufWriteStr (BENZINA_BUF*  bbuf, const char*        str){
	return benzinaBufWriteLDL(bbuf, str, strlen(str));
}
BENZINA_HIDDEN int          benzinaBufWriteMsg (BENZINA_BUF*  bbuf,
                                                const BENZINA_BUF* msg,
                                                size_t        len){
	return benzinaBufWriteLDL(bbuf, msg->buf+msg->off, len);
}

BENZINA_HIDDEN int          benzinaBufWritevu64(BENZINA_BUF*  bbuf, uint64_t  v){
	char p[10], *q = p;
	
	*q  = v & 0x7F;
	v >>= 7;
	while(v){
		*q++ |= 0x80;
		*q    = v & 0x7F;
		v   >>= 7;
	}
	
	return benzinaBufWrite(bbuf, p, q-p+1);
}
BENZINA_HIDDEN int          benzinaBufWritevs64(BENZINA_BUF*  bbuf, int64_t   v){
	return benzinaBufWritevu64(bbuf, (v<<1) ^ (v>>63));
}
BENZINA_HIDDEN int          benzinaBufWritevi64(BENZINA_BUF*  bbuf, int64_t   v){
	return benzinaBufWritevu64(bbuf, v);
}
BENZINA_HIDDEN int          benzinaBufWritevu32(BENZINA_BUF*  bbuf, uint32_t  v){
	return benzinaBufWritevu64(bbuf, v & (uint32_t)~0);
}
BENZINA_HIDDEN int          benzinaBufWritevs32(BENZINA_BUF*  bbuf, int32_t   v){
	return benzinaBufWritevu32(bbuf, (v<<1) ^ (v>>31));
}
BENZINA_HIDDEN int          benzinaBufWritevi32(BENZINA_BUF*  bbuf, int32_t   v){
	return benzinaBufWritevu32(bbuf, v);
}

BENZINA_HIDDEN int          benzinaBufWritefu64(BENZINA_BUF*  bbuf, uint64_t  u){
	return benzinaBufWrite(bbuf, (char*)&u, sizeof(u));
}
BENZINA_HIDDEN int          benzinaBufWritefs64(BENZINA_BUF*  bbuf, int64_t   s){
	return benzinaBufWrite(bbuf, (char*)&s, sizeof(s));
}
BENZINA_HIDDEN int          benzinaBufWriteff64(BENZINA_BUF*  bbuf, double    f){
	return benzinaBufWrite(bbuf, (char*)&f, sizeof(f));
}
BENZINA_HIDDEN int          benzinaBufWritefu32(BENZINA_BUF*  bbuf, uint32_t  u){
	return benzinaBufWrite(bbuf, (char*)&u, sizeof(u));
}
BENZINA_HIDDEN int          benzinaBufWritefs32(BENZINA_BUF*  bbuf, int32_t   s){
	return benzinaBufWrite(bbuf, (char*)&s, sizeof(s));
}
BENZINA_HIDDEN int          benzinaBufWriteff32(BENZINA_BUF*  bbuf, float     f){
	return benzinaBufWrite(bbuf, (char*)&f, sizeof(f));
}

BENZINA_HIDDEN int          benzinaBufWriteEnum(BENZINA_BUF*  bbuf, int32_t   e){
	return benzinaBufWritevi32(bbuf, e);
}
BENZINA_HIDDEN int          benzinaBufWriteBool(BENZINA_BUF*  bbuf, uint64_t  b){
	return benzinaBufWritevi32(bbuf, !!b);
}

BENZINA_HIDDEN int          benzinaBufWriteTagW(BENZINA_BUF*  bbuf,
                                                uint32_t      tag,
                                                uint32_t      wire){
	return benzinaBufWritevi32(bbuf, (tag << 3) | (wire & 0x7));
}
