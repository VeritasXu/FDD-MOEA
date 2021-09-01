import numpy as np
from pymoo.model.problem import Problem


class Multi_AF(Problem):
    def __init__(self, server, clients, ac_type='G', n_var=30, n_obj=2, xl=0, xu=1, **kwargs):
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=0, xl=xl, xu=xu, **kwargs)
        self.server = server
        self.clients = clients
        self.ac_type = ac_type

    def _calc_pareto_front(self, n_pareto_points=100):
        return None

    def _evaluate(self, x, out, *args, **kwargs):

        if self.clients is not None:
            num_c = len(self.clients)
            fl_hat = np.zeros((num_c, x.shape[0], self.n_obj))

            for i in range(num_c):
                fl_hat[i, :, :] = self.clients[i].predict(x)

            client_pred = np.mean(fl_hat, axis=0)
        else:
            num_c = 0
            fl_hat = None
            client_pred = None

        server_pred = self.server.predict(x)

        if self.ac_type == 'L':
            f_mean = client_pred
            tmp = (fl_hat - f_mean) ** 2
            s_2_hat = np.mean(tmp, axis=0) * num_c / (num_c - 1)
            s_hat = np.sqrt(s_2_hat)

        elif self.ac_type == 'G':
            f_mean = server_pred
            tmp = (fl_hat - f_mean) ** 2
            s_2_hat = np.mean(tmp, axis=0) * num_c / (num_c - 1)
            s_hat = np.sqrt(s_2_hat)

        elif self.ac_type == 'LG':
            f_mean = (client_pred + server_pred) / 2

            combined_f_hat = np.insert(fl_hat, -1, server_pred, axis=0)

            tmp = (combined_f_hat - f_mean) ** 2

            s_2_hat = np.mean(tmp, axis=0) * (num_c + 1) / num_c

            s_hat = np.sqrt(s_2_hat)

        else:
            f_mean = None
            s_hat = None
            print('Error in acquisition function type!')

        w = 2
        LCB_matrix = f_mean - w * s_hat

        out["F"] = LCB_matrix
