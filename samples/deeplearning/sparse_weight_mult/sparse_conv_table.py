import os

sparse_fac = 0.5


print('W/H bigger then 16')
print('R=1')
for w in [28,56]:
    for c in [64,128]:
        for k in [64,128]:
            for n in [64,128]:
                stream = os.popen('./run_conv.sh {c} {k} {n} {sparse_fac} 100 {w} 1 1'.format(c=c,k=k,n=n,sparse_fac=sparse_fac,w=w))
                output = stream.read()
                out_ind_1 = output.find('FWD_MAX_PERF')
                out_ind_2 = output.find('GFLOPS')
                print(output[out_ind_1:out_ind_2+6])

print('W/H bigger then 16')
print('R=5')
for w in [28,56]:
    for c in [64,128]:
        for k in [64,128]:
            for n in [64,128]:
                stream = os.popen('./run_conv.sh {c} {k} {n} {sparse_fac} 100 {w} 5 1'.format(c=c,k=k,n=n,sparse_fac=sparse_fac,w=w))
                output = stream.read()
                out_ind_1 = output.find('FWD_MAX_PERF')
                out_ind_2 = output.find('GFLOPS')
                print(output[out_ind_1:out_ind_2+6])


print('W/H smaller then 16')
print('R=1')
for w in [7,14]:
    for c in [64,128]:
        for k in [64,128]:
            for n in [64,128]:
                stream = os.popen('./run_conv.sh {c} {k} {n} {sparse_fac} 100 {w} 1 1'.format(c=c,k=k,n=n,sparse_fac=sparse_fac,w=w))
                output = stream.read()
                out_ind_1 = output.find('FWD_MAX_PERF')
                out_ind_2 = output.find('GFLOPS')
                print(output[out_ind_1:out_ind_2+6])

print('W/H smaller then 16')
print('R=5')
for w in [7,14]:
    for c in [64,128]:
        for k in [64,128]:
            for n in [64,128]:
                stream = os.popen('./run_conv.sh {c} {k} {n} {sparse_fac} 100 {w} 5 1'.format(c=c,k=k,n=n,sparse_fac=sparse_fac,w=w))
                output = stream.read()
                out_ind_1 = output.find('FWD_MAX_PERF')
                out_ind_2 = output.find('GFLOPS')
                print(output[out_ind_1:out_ind_2+6])
