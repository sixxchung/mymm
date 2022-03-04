xmlFile= '/home/oschung_skcc/git/mmdetection/my/input_image/oxford/annotations/xmls/Abyssinian_115.xml'

from lxml import etree
x = etree.parse(xmlFile)
pretty_xml = etree.tostring(x, pretty_print=True, encoding=str)
print(pretty_xml)

def foo(arg1,arg2):
    #do something with args
    a = arg1 + arg2
    return a

import inspect

print( inspect.getsource(foo) )
# def foo(arg1,arg2):
#     #do something with args
#     a = arg1 + arg2
#     return a