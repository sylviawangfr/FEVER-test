import xmltodict
from memory_profiler import profile
import urllib.request
import gc

@profile
def ip():
    r = urllib.request.urlopen('http://localhost:5005/lookup-application/api/search?label=USA')
    b = r.read().decode('utf-8')
    # x = xmltodict.parse(b)
    r.close()
    # del b
    # print(len(b))
    return b


# @profile
def test1():
    for i in range(30):
        result = ip()
        del result
        gc.collect()


@profile
def test2():
    x = '''<?xml version="1.0" encoding="utf-8"?> 
    <ArrayOfResult xmlns="http://lookup.dbpedia.org/" 
    xmlns:xsd="http://www.w3.org/2001/XMLSchema" 
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
    </ArrayOfResult>'''
    for i in range(50):
        t = xmltodict.parse(x)
        del t
        gc.collect()

    for i in range(50):
        t = xmltodict.parse(x)
        del t
        gc.collect()

    return

test1()