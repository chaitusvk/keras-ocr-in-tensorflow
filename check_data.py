from data import Data

step = 10000

data = Data()

for i in range(step):

	data.next_batch(8,64)
	print(i)
