import numpy as np


def variable_file(firsttrain, callbackperiod, maxepochs, variablefilename):
    import csv
    import math

    def get_variable_f(v, var):
        low, up = v * 0.2, v * 1.8
        l = (up - low) / 2
        v1 = l * math.tanh(var) + l + low
        return v1

    f = open(variablefilename)
    csv_f = csv.reader(f)

    _E = []
    _ti = []
    _tp = []
    _td = []
    _k = []
    _Rm = []
    _a1 = []
    _C1 = []
    _C2 = []
    _C4 = []
    _C5 = []
    _Ub = []
    _U0 = []
    _Um = []
    _Rg = []
    _alpha = []
    _beta = []

    counter = firsttrain
    callbackcount = []

    while counter <= maxepochs + firsttrain:
        callbackcount.append(str(counter) + " [")
        counter = counter + callbackperiod

    for row in csv_f:
        _E.append(row[0])
        _ti.append(row[1])
        _tp.append(row[2])
        _td.append(row[3])
        _k.append(row[4])
        _Rm.append(row[5])
        _a1.append(row[6])
        _C1.append(row[7])
        _C2.append(row[8])
        _C4.append(row[9])
        _C5.append(row[10])
        _Ub.append(row[11])
        _U0.append(row[12])
        _Um.append(row[13])
        _Rg.append(row[14])
        _alpha.append(row[15])
        _beta.append(row[16])

    f.close()

    count = 0
    for i in _E:
        _E[count] = i.replace(callbackcount[count], "")
        count = count + 1

    _beta = [s.replace("]", "") for s in _beta]

    _E = [float(i) for i in _E]
    _ti = [float(i) for i in _ti]
    _tp = [float(i) for i in _tp]
    _td = [float(i) for i in _td]
    _k = [float(i) for i in _k]
    _Rm = [float(i) for i in _Rm]
    _a1 = [float(i) for i in _a1]
    _C1 = [float(i) for i in _C1]
    _C2 = [float(i) for i in _C2]
    _C4 = [float(i) for i in _C4]
    _C5 = [float(i) for i in _C5]
    _Ub = [float(i) for i in _Ub]
    _U0 = [float(i) for i in _U0]
    _Um = [float(i) for i in _Um]
    _Rg = [float(i) for i in _Rg]
    _alpha = [float(i) for i in _alpha]
    _beta = [float(i) for i in _beta]

    E = [(math.tanh(n) + 1) * 0.1 + 0.1 for n in _E]
    tp = [(math.tanh(n) + 1) * 2 + 4 for n in _tp]
    ti = [(math.tanh(n) + 1) * 40 + 60 for n in _ti]
    td = [(math.tanh(n) + 1) * 25 / 6 + 25 / 3 for n in _td]
    k = [get_variable_f(0.0083, n) for n in _k]
    Rm = [get_variable_f(209 / 100, n) * 100 for n in _Rm]
    a1 = [get_variable_f(6.6, n) for n in _a1]
    C1 = [get_variable_f(300 / 100, n) * 100 for n in _C1]
    C2 = [get_variable_f(144 / 100, n) * 100 for n in _C2]
    C4 = [get_variable_f(80 / 100, n) * 100 for n in _C4]
    C5 = [get_variable_f(26 / 100, n) * 100 for n in _C5]
    Ub = [get_variable_f(72 / 100, n) * 100 for n in _Ub]
    U0 = [get_variable_f(4 / 100, n) * 100 for n in _U0]
    Um = [get_variable_f(90 / 100, n) * 100 for n in _Um]
    Rg = [get_variable_f(180 / 100, n) * 100 for n in _Rg]
    alpha = [get_variable_f(7.5, n) for n in _alpha]
    beta = [get_variable_f(1.772, n) for n in _beta]

    callbackcount = [s.replace(" [", "") for s in callbackcount]
    var_list = [
        callbackcount,
        E,
        tp,
        ti,
        td,
        k,
        Rm,
        a1,
        C1,
        C2,
        C4,
        C5,
        Ub,
        U0,
        Um,
        Rg,
        alpha,
        beta,
    ]
    var_list_names = [
        "Epoch",
        "E",
        "tp",
        "ti",
        "td",
        "k",
        "Rm",
        "a1",
        "C1",
        "C2",
        "C4",
        "C5",
        "Ub",
        "U0",
        "Um",
        "Rg",
        "alpha",
        "beta",
    ]

    with open("varlist.csv", "w", newline="") as datafile:
        writer = csv.writer(datafile)
        writer.writerow(var_list_names)
        writer.writerows(np.transpose(var_list))
        datafile.close()
