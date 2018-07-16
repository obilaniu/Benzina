import sys, torch, benzina.torch.nvdecode as B, time
x = torch.cuda.FloatTensor(10,10)
d = B.BenzinaDataset(sys.argv[1])
l = B.BenzinaLoader(d, 256)
n = 0
try:
	t =- time.time()
	for s in l:
		n += len(s)
except:
	pass
finally:
	t += time.time()
	print("Time:   {}".format(t))
	print("Images: {}".format(n))
	print("Speed:  {} images/second".format(n/t))
