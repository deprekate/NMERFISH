#!/usr/bin/env python3

import os
import sys
import argparse
import traceback

import numpy as np
import zarr
from zarr.errors import PathNotFoundError
from numcodecs import BZ2

#z0 = zarr.open('data', mode='r')
def is_zarr(x):
	try:
		z = zarr.open(os.path.join(x,'data'), mode='r')
		return z
	except PathNotFoundError as e:
		traceback.print_exc()
		raise

def not_zarr(x):
	if os.path.exists(x):
		raise argparse.ArgumentTypeError("{0} already exists".format(x))
	return x

if __name__ == '__main__':
	usage = '%s [-opt1, [-opt2, ...]] inzarr outzarr' % __file__
	parser = argparse.ArgumentParser(description='', formatter_class=argparse.RawTextHelpFormatter, usage=usage)
	parser.add_argument('inzarr', type=is_zarr, help='input zarr')
	parser.add_argument('outzarr',type=not_zarr, help='where to write the recast zarr')
	parser.add_argument('-m', '--method', help='The method for downscaling the inzarr', type=str, default='sqrt', choices=['sqrt', 'divi'])
	args = parser.parse_args()
	
	bz2_compressor = BZ2(level=9)
	shape = args.inzarr.shape
	chunks = args.inzarr.chunks

	# Create the Zarr array
	os.mkdir(args.outzarr)
	store = zarr.DirectoryStore('%s/data' % args.outzarr)
	z = zarr.create(store=store, shape=shape, chunks=chunks, dtype='|u1', 
					fill_value=None, compressor=bz2_compressor,overwrite=False)
	z.attrs['downscale'] = args.method

	# Iterate over all chunks in the original Zarr array
	for i in range(shape[0]):
		for j in range(shape[1] // chunks[1]):
			for k in range(shape[2] // chunks[2]):
				chunk_slices = (
					slice(i * chunks[0], min((i + 1) * chunks[0], shape[0])),
					slice(j * chunks[1], min((j + 1) * chunks[1], shape[1])),
					slice(k * chunks[2], min((k + 1) * chunks[2], shape[2]))
				)
				#z1[chunk_slices] = z0[chunk_slices] // 256
				z[chunk_slices] = np.sqrt(args.inzarr[chunk_slices]).astype(np.uint8)

	store.close()

