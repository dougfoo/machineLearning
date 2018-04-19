
import sys,re

STARTMARKER='Usage of azlyrics.com content by any third-party lyrics provider is prohibited by our licensing agreement. Sorry about that'
ENDMARKER='<!-- MxM banner -->'
indata=sys.stdin.readlines()
start_line=0
end_line=0
for i in range(len(indata)):
    if STARTMARKER in indata[i]:
        start_line=i
    if ENDMARKER in indata[i]:
        end_line=i

indata=indata[start_line-1:end_line-2]

for i in range(len(indata)): indata[i]=re.sub(r'<.*>','',indata[i].strip())
    

for i in indata:
    print i
