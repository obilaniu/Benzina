#!/usr/bin/env python

#
# Import sundry Python packages
#

import os
import os.path
import subprocess
import sys

# Important to locate ourselves.
basename = os.path.basename(sys.argv[0])
dirname  = os.path.dirname(sys.argv[0])


#
# Symlink oneself into the dirname/.git/hooks folder, if needed.
#

def install_hooks():
	install_hook("pre-commit")

def install_hook(name):
	linkPath = dirname+"/.git/hooks/"+name
	if os.path.exists(linkPath):
		os.unlink(linkPath)
	os.symlink("../../install-hooks.py", linkPath)

#
# Run pre-commit check.
#

def pre_commit():
	# Get commit against which to check.
	if subprocess.call("git rev-parse --verify HEAD >/dev/null 2>&1", shell=True) == 0:
		against="HEAD"
	else:
		against="4b825dc642cb6eb9a060e54bf8d69288fbee4904" # Empty object
	
	# Get list of changed files.
	fList = subprocess.check_output("git diff --cached --name-only --diff-filter=AM {:s}".format(against), shell=True)
	
	# Get threshold file size from config hook.maxfilesize, or else use 1MB.
	try:
		threshStr = subprocess.check_output("git config --get hook.maxfilesize".format(against), shell=True)
	except:
		threshStr = ""
	if(threshStr == ""):
		threshStr = "1M"
	if threshStr[-1].upper() in "KMG":
		sizeThreshold = int(1024**({"K":1,"M":2,"G":3}[threshStr[-1].upper()]))
	elif not threshStr[-1].isdigit():
		print("Error: git config hook.maxfilesize is set to invalid value "+threshStr+" !")
		sys.exit(1)
	else:
		sizeThreshold = 1
	sizeThreshold = sizeThreshold * int(threshStr[:-1])
	
	# Check all files:
	fErrors = 0
	for fName in fList.splitlines():
		# Absolutely prevent .pyc files from being committed.
		if fName.endswith(".pyc"):
			fErrors += 1
			print("File '{:s}' is a Python interpreter-generated file that should not be committed!".format(fName))
		
		# Prevent oversize files from being committed.
		size = os.stat(fName).st_size
		if size > sizeThreshold:
			fErrors += 1
			print("File '{:s}' is {:d} bytes > {:s}!".format(fName, size, threshStr))
	
	# Exit.
	sys.exit(0 if fErrors==0 else 1)



#
# Main
#

if __name__ == "__main__":
	if   basename == "install-hooks.py":
		install_hooks()
	elif sys.argv[0] == '.git/hooks/pre-commit':
		pre_commit()
	else:
		pass
	
	sys.exit(0)

