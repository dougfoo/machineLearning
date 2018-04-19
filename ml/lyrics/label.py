
import os

def label_examples():
    doclist=[]
    for root,dir,file in os.walk('.'):
        if root=='.': continue
        root=root.lstrip('./')
        for i in file:
            node=(root,root+'/'+i)
            doclist.append(node)
            
    return doclist



