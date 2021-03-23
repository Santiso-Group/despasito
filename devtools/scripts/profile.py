import despasito
import sys

input_fname, path = sys.argv[1:]

args = {"filename": input_fname, "path": path}

despasito.run(**args)
