import sys

import numpy
import mlflow
import os.path

Ri_cols_max = []
Ri_cols_min = []
Ri_rows_max = []
Ri_rows_min = []
Ro_cols_max = []
Ro_cols_min = []
Ro_rows_max = []
Ro_rows_min = []
X_max = []
X_min = []
y_max = []
y_min = []


count = 1000
input_dir = sys.argv[1]
merged_output_dir = sys.argv[2]

with mlflow.start_run() as run:
    while count < 10000:

        if count % 500 == 0:
            print('Merging Event: '+str(count))

        a = numpy.load(os.path.join(input_dir, 'event00000'+str(count)+'_g000.npz'))
        b = numpy.load(os.path.join(input_dir, 'event00000'+str(count)+'_g001.npz'))
        c = numpy.load(os.path.join(input_dir, 'event00000'+str(count)+'_g002.npz'))
        d = numpy.load(os.path.join(input_dir, 'event00000'+str(count)+'_g003.npz'))
        e = numpy.load(os.path.join(input_dir, 'event00000'+str(count)+'_g004.npz'))
        f = numpy.load(os.path.join(input_dir, 'event00000'+str(count)+'_g005.npz'))
        g = numpy.load(os.path.join(input_dir, 'event00000'+str(count)+'_g006.npz'))
        h = numpy.load(os.path.join(input_dir, 'event00000'+str(count)+'_g007.npz'))

        Ri_cols = [*a['Ri_cols'],*b['Ri_cols'],*c['Ri_cols'],*d['Ri_cols'],*e['Ri_cols'],*f['Ri_cols'],*g['Ri_cols'],*h['Ri_cols']]
        Ri_rows = [*a['Ri_rows'],*b['Ri_rows'],*c['Ri_rows'],*d['Ri_rows'],*e['Ri_rows'],*f['Ri_rows'],*g['Ri_rows'],*h['Ri_rows']]
        Ro_cols = [*a['Ro_cols'],*b['Ro_cols'],*c['Ro_cols'],*d['Ro_cols'],*e['Ro_cols'],*f['Ro_cols'],*g['Ro_cols'],*h['Ro_cols']]
        Ro_rows = [*a['Ro_rows'],*b['Ro_rows'],*c['Ro_rows'],*d['Ro_rows'],*e['Ro_rows'],*f['Ro_rows'],*g['Ro_rows'],*h['Ro_rows']]
        X = [*a['X'],*b['X'],*c['X'],*d['X'],*e['X'],*f['X'],*g['X'],*h['X']]
        y = [*a['y'],*b['y'],*c['y'],*d['y'],*e['y'],*f['y'],*g['y'],*h['y']]

        #print('Ri_cols  max:'+str(max(Ri_cols))+' min:'+str(min(Ri_cols)))
        #print('Ri_rows  max:'+str(max(Ri_rows))+' min:'+str(min(Ri_rows)))
        #print('Ro_cols  max:'+str(max(Ro_cols))+' min:'+str(min(Ro_cols)))
        #print('Ro_rows  max:'+str(max(Ro_rows))+' min:'+str(min(Ro_rows)))
        #print('X  max:'+str(numpy.amax(X))+' min:'+str(numpy.amin(X)))
        #print('y  max:'+str(max(y))+' min:'+str(min(y)))

        Ri_cols_max.append(max(Ri_cols))
        Ri_cols_min.append(min(Ri_cols))
        Ri_rows_max.append(max(Ri_rows))
        Ri_rows_min.append(min(Ri_rows))
        Ro_cols_max.append(max(Ro_cols))
        Ro_cols_min.append(min(Ro_cols))
        Ro_rows_max.append(max(Ro_rows))
        Ro_rows_min.append(min(Ro_rows))
        X_max.append(numpy.amax(X))
        X_min.append(numpy.amin(X))
        y_max.append(max(y))
        y_min.append(min(y))

        numpy.savez(os.path.join(merged_output_dir, 'event00000' + str(count) + '.npz'),
                    Ri_cols=Ri_cols, Ri_rows=Ri_rows, Ro_cols=Ro_cols,
                    Ro_rows=Ro_rows, X=X, y=y)

        if count == 2399:
            count = 2450
        elif count == 5899:
            count = 5950
        elif count == 8699:
            count = 8750
        else:
            count = count + 1

    print('Ri_cols  max:'+str(max(Ri_cols_max))+' min:'+str(min(Ri_cols_min)))
    print('Ri_rows  max:'+str(max(Ri_rows_max))+' min:'+str(min(Ri_rows_min)))
    print('Ro_cols  max:'+str(max(Ro_cols_max))+' min:'+str(min(Ro_cols_min)))
    print('Ro_rows  max:'+str(max(Ro_rows_max))+' min:'+str(min(Ro_rows_min)))
    print('X  max:'+str(max(X_max))+' min:'+str(min(X_min)))
    print('y  max:'+str(max(y_max))+' min:'+str(min(y_min)))



