--[[
SipHash tests, using the Test Anything Protocol (TAP)
--]]

print('1..6')


-- Small utility function to save typing while defining tests
function tstmessagetap(name, cond)
  if cond then print('ok '..name)
  else         print('not ok '..name)
  end
end



--[[
SipHash test value

The following is from the worked example from Appendix A of the SipHash paper [1]

    [1] https://www.aumasson.jp/siphash/siphash.pdf
--]]

siphash = require 'benzina.siphash'.siphash

data = '\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e'
key  = '\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f'
k0   = 0x0706050403020100 -- 1st half, little-endian 64-bit
k1   = 0x0f0e0d0c0b0a0908 -- 2nd half, little-endian 64-bit
hash = 0xa129ca6149be45e5

tstmessagetap('siphash 1 (int key, default c,d)',
              hash == siphash(data, k0, k1))
tstmessagetap('siphash 2 (int key, c=2, default d)',
              hash == siphash(data, k0, k1, 2))
tstmessagetap('siphash 3 (int key, c=2, d=4)',
              hash == siphash(data, k0, k1, 2, 4))
tstmessagetap('siphash 4 (string key, default c,d)',
              hash == siphash(data, key))
tstmessagetap('siphash 5 (string key, c=2, default d)',
              hash == siphash(data, key, 2))
tstmessagetap('siphash 6 (string key, c=2, d=4)',
              hash == siphash(data, key, 2, 4))
