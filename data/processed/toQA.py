with open('./cornell_train.txt','r') as f:
	lines=f.readlines()
	with open('./QA.txt','w') as w:	
		for i in range(len(lines)):
			if i%2==0:	
				w.write(lines[i].strip()+'|!|')

			else:
				w.write(lines[i])


