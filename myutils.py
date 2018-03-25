import time, itertools

def time_fn( fn, *args, **kwargs ):
    start = time.clock()
    results = fn( *args, **kwargs )
    end = time.clock()
    #fn_name = fn.__module__ + "." + fn.__name__
    #print fn_name + ": " + str(end-start) + "s"
    return [results,end-start]

def churn(d, n):
    for _ in itertools.repeat(None, n):
        d = d.append(d)
    return d

