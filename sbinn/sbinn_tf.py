import numpy as np
import deepxde as dde
from deepxde.backend import tf
import variable_to_parameter_transform


def sbinn(data_t, data_y, meal_t, meal_q):
    def get_variable(v, var):
        low, up = v * 0.2, v * 1.8
        l = (up - low) / 2
        v1 = l * tf.tanh(var) + l + low
        return v1

    E_ = dde.Variable(0.0)
    tp_ = dde.Variable(0.0)
    ti_ = dde.Variable(0.0)
    td_ = dde.Variable(0.0)
    k_ = dde.Variable(0.0)
    Rm_ = dde.Variable(0.0)
    a1_ = dde.Variable(0.0)
    C1_ = dde.Variable(0.0)
    C2_ = dde.Variable(0.0)
    C4_ = dde.Variable(0.0)
    C5_ = dde.Variable(0.0)
    Ub_ = dde.Variable(0.0)
    U0_ = dde.Variable(0.0)
    Um_ = dde.Variable(0.0)
    Rg_ = dde.Variable(0.0)
    alpha_ = dde.Variable(0.0)
    beta_ = dde.Variable(0.0)

    var_list_ = [
        E_,
        tp_,
        ti_,
        td_,
        k_,
        Rm_,
        a1_,
        C1_,
        C2_,
        C4_,
        C5_,
        Ub_,
        U0_,
        Um_,
        Rg_,
        alpha_,
        beta_,
    ]

    def ODE(t, y):
        Ip = y[:, 0:1]
        Ii = y[:, 1:2]
        G = y[:, 2:3]
        h1 = y[:, 3:4]
        h2 = y[:, 4:5]
        h3 = y[:, 5:6]

        Vp = 3
        Vi = 11
        Vg = 10
        E = (tf.tanh(E_) + 1) * 0.1 + 0.1
        tp = (tf.tanh(tp_) + 1) * 2 + 4
        ti = (tf.tanh(ti_) + 1) * 40 + 60
        td = (tf.tanh(td_) + 1) * 25 / 6 + 25 / 3
        k = get_variable(0.0083, k_)
        Rm = get_variable(209, Rm_)
        a1 = get_variable(6.6, a1_)
        C1 = get_variable(300, C1_)
        C2 = get_variable(144, C2_)
        C3 = 100
        C4 = get_variable(80, C4_)
        C5 = get_variable(26, C5_)
        Ub = get_variable(72, Ub_)
        U0 = get_variable(4, U0_)
        Um = get_variable(90, Um_)
        Rg = get_variable(180, Rg_)
        alpha = get_variable(7.5, alpha_)
        beta = get_variable(1.772, beta_)

        f1 = Rm * tf.math.sigmoid(G / (Vg * C1) - a1)
        f2 = Ub * (1 - tf.math.exp(-G / (Vg * C2)))
        kappa = (1 / Vi + 1 / (E * ti)) / C4
        f3 = (U0 + Um / (1 + tf.pow(tf.maximum(kappa * Ii, 1e-3), -beta))) / (Vg * C3)
        f4 = Rg * tf.sigmoid(alpha * (1 - h3 / (Vp * C5)))
        dt = t - meal_t
        IG = tf.math.reduce_sum(
            0.5 * meal_q * k * tf.math.exp(-k * dt) * (tf.math.sign(dt) + 1),
            axis=1,
            keepdims=True,
        )
        tmp = E * (Ip / Vp - Ii / Vi)
        dIP_dt = dde.grad.jacobian(y, t, i=0, j=0)
        dIi_dt = dde.grad.jacobian(y, t, i=1, j=0)
        dG_dt = dde.grad.jacobian(y, t, i=2, j=0)
        dh1_dt = dde.grad.jacobian(y, t, i=3, j=0)
        dh2_dt = dde.grad.jacobian(y, t, i=4, j=0)
        dh3_dt = dde.grad.jacobian(y, t, i=5, j=0)
        return [
            dIP_dt - (f1 - tmp - Ip / tp),
            dIi_dt - (tmp - Ii / ti),
            dG_dt - (f4 + IG - f2 - f3 * G),
            dh1_dt - (Ip - h1) / td,
            dh2_dt - (h1 - h2) / td,
            dh3_dt - (h2 - h3) / td,
        ]

    geom = dde.geometry.TimeDomain(data_t[0, 0], data_t[-1, 0])

    # Observes
    n = len(data_t)
    idx = np.append(
        np.random.choice(np.arange(1, n - 1), size=n // 5, replace=False), [0, n - 1]
    )
    observe_y2 = dde.PointSetBC(data_t[idx], data_y[idx, 2:3], component=2)

    np.savetxt("glucose_input.dat", np.hstack((data_t[idx], data_y[idx, 2:3])))

    data = dde.data.PDE(geom, ODE, [observe_y2], anchors=data_t)

    net = dde.maps.FNN([1] + [128] * 3 + [6], "swish", "Glorot normal")

    def feature_transform(t):
        t = 0.01 * t
        return tf.concat(
            (t, tf.sin(t), tf.sin(2 * t), tf.sin(3 * t), tf.sin(4 * t), tf.sin(5 * t)),
            axis=1,
        )

    net.apply_feature_transform(feature_transform)

    def output_transform(t, y):
        idx = 1799
        k = (data_y[idx] - data_y[0]) / (data_t[idx] - data_t[0])
        b = (data_t[idx] * data_y[0] - data_t[0] * data_y[idx]) / (
            data_t[idx] - data_t[0]
        )
        linear = k * t + b
        factor = tf.math.tanh(t) * tf.math.tanh(idx - t)
        return linear + factor * tf.constant([1, 1, 1e2, 1, 1, 1]) * y

    net.apply_output_transform(output_transform)

    model = dde.Model(data, net)

    firsttrain = 10000
    callbackperiod = 1000
    maxepochs = 1000000

    model.compile("adam", lr=1e-3, loss_weights=[0, 0, 0, 0, 0, 0, 1e-2])
    model.train(epochs=firsttrain, display_every=1000)
    model.compile(
        "adam",
        lr=1e-3,
        loss_weights=[1, 1, 1e-2, 1, 1, 1, 1e-2],
        external_trainable_variables=var_list_,
    )
    variablefilename = "variables.csv"
    variable = dde.callbacks.VariableValue(
        var_list_, period=callbackperiod, filename=variablefilename
    )
    losshistory, train_state = model.train(
        epochs=maxepochs, display_every=1000, callbacks=[variable]
    )

    dde.saveplot(losshistory, train_state, issave=True, isplot=True)


gluc_data = np.hsplit(np.loadtxt("glucose.dat"), [1])
meal_data = np.hsplit(np.loadtxt("meal.dat"), [4])

t = gluc_data[0]
y = gluc_data[1]
meal_t = meal_data[0]
meal_q = meal_data[1]

sbinn(
    t[:1800],
    y[:1800],
    meal_t,
    meal_q,
)

variable_to_parameter_transform.variable_file(10000, 1000, 1000000, "variables.csv")
