from time import time


def timer(func):
	def wrapper(*args, **kwargs):
		start = time()
		rtrn = func(*args, **kwargs)
		end = time()
		print("%r %.3f s" % (func.__name__, end - start))
		return rtrn
	return wrapper


