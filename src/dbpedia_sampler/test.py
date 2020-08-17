import xmltodict
from memory_profiler import profile


# @profile
def test():
    x = '''<?xml version="1.0" encoding="utf-8"?> 
    <ArrayOfResult xmlns="http://lookup.dbpedia.org/" 
    xmlns:xsd="http://www.w3.org/2001/XMLSchema" 
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
    </ArrayOfResult>'''
    for i in range(30):
        t = xmltodict.parse(x)
        del t

    for i in range(30):
        t = xmltodict.parse(x)
        del t

    return

test()