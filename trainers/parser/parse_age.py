import sys
program_name = sys.argv[0]
arguments = sys.argv[1:]

lines = [line.rstrip('\n') for line in open(arguments[0])]
del lines[0]
total = 0
correct = 0
for l in lines:
        total += 1
        p = l.split(",")
        a = p[0].split("/")[0]
        b = p[1]
        correct += abs(int(a)-int(b))

print(correct / total)
~                        