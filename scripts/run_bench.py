import subprocess, sys, os

for i in range(0, 52):
	cmds = ["ffmpeg -y -i ../test.png -pix_fmt yuv420p -crf {:d} test-h264-crf-{:02d}.h264 1>/dev/null 2>&1".format(i,i),
	        "ffmpeg -y -i test-h264-crf-{:02d}.h264 test-h264-crf-{:02d}.png 1>/dev/null 2>&1".format(i,i),
	        "python /home/wotan/Documents/Software/Sources/Git/CodeSnips/python/pysnips/psnrhma.py ../test.png test-h264-crf-{:02d}.png".format(i)]
	subprocess.Popen(cmds[0], shell=True, stdout=None, stderr=None).wait()
	subprocess.Popen(cmds[1], shell=True, stdout=None, stderr=None).wait()
	sys.stdout.write("H264    {:02d}      {:18d}    ".format(i, os.stat("test-h264-crf-{:02d}.h264".format(i)).st_size))
	sys.stdout.flush()
	subprocess.Popen(cmds[2], shell=True, stdout=None, stderr=None).wait()

for i in range(0, 52):
    cmds = ["ffmpeg -y -i ../test.png -pix_fmt yuv420p -crf {:d} test-h265-crf-{:02d}.h265 1>/dev/null 2>&1".format(i,i),
            "ffmpeg -y -i test-h265-crf-{:02d}.h265 test-h265-crf-{:02d}.png 1>/dev/null 2>&1".format(i,i),
            "python /home/wotan/Documents/Software/Sources/Git/CodeSnips/python/pysnips/psnrhma.py ../test.png test-h265-crf-{:02d}.png".format(i)]
    subprocess.Popen(cmds[0], shell=True, stdout=None, stderr=None).wait()
    subprocess.Popen(cmds[1], shell=True, stdout=None, stderr=None).wait()
    sys.stdout.write("H265    {:02d}      {:18d}    ".format(i, os.stat("test-h265-crf-{:02d}.h265".format(i)).st_size))
    sys.stdout.flush()
    subprocess.Popen(cmds[2], shell=True, stdout=None, stderr=None).wait()

for i in range(1, 96)[::-1]:
	cmds = ["convert ../test.png -quality {:d} test-jpeg-qf-{:02d}.jpeg 1>/dev/null 2>&1".format(i,i),
	        "ffmpeg -y -i test-jpeg-qf-{:02d}.jpeg test-jpeg-qf-{:02d}.png 1>/dev/null 2>&1".format(i,i),
	        "python /home/wotan/Documents/Software/Sources/Git/CodeSnips/python/pysnips/psnrhma.py ../test.png test-jpeg-qf-{:02d}.png".format(i)]
	subprocess.Popen(cmds[0], shell=True, stdout=None, stderr=None).wait()
	subprocess.Popen(cmds[1], shell=True, stdout=None, stderr=None).wait()
	sys.stdout.write("JPEG    {:02d}      {:18d}    ".format(i, os.stat("test-jpeg-qf-{:02d}.jpeg".format(i)).st_size))
	sys.stdout.flush()
	subprocess.Popen(cmds[2], shell=True, stdout=None, stderr=None).wait()

