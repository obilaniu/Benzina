/**
 * FILE DESCRIPTOR CACHING:
 * 
 * Done in Python, by fstat()+fstatfs()+fcntl(F_GETFL) on open fds.
 * 
 * 
 * MP4 FILE OPENING:
 * 
 * Can create by path or by fd.
 * 
 * If by path, can have external dref.
 *   - Open strategy is to chop off but keep dirname, then
 *     openat(dirname, basename).
 * If by fd, must be self-contained.
 */
