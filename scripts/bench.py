import sys, time, torch, pdb
import benzina.torch as B


if __name__ == "__main__":
	x = torch.cuda.FloatTensor(10,10)
	del x
	d = B.ImageNet(sys.argv[1])
	l = B.NvdecodeDataLoader(d,
	                         batch_size      = 256,
	                         seed            = 0,
	                         shape           = (256,256),
	                         warp_transform  = None,
	                         oob_transform   = (0,0,0),
	                         scale_transform = 1/255,
	                         bias_transform  = -0.5)
	n = 0
	try:
		t =- time.time()
		for images, targets in l:
			#
			# The targets tensor is still collated on CPU. Move it to same
			# device as images.
			#
			targets = targets.to(images.device)
			n += len(images)
	except:
		raise
	finally:
		t += time.time()
		print("Time:   {}".format(t))
		print("Images: {}".format(n))
		print("Speed:  {} images/second".format(n/t))
