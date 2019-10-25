import re

tmp=re.match("\d+_(\d+)\.txt", '324_324.txt').group(1)

print(tmp)