# %%
import os
import sys
import time

import numpy as np
import pandas as pd
import pyreadr
import torch as tch
from bayes_opt import BayesianOptimization
from bayes_opt.acquisition import ExpectedImprovement
from sklearn.preprocessing import StandardScaler

# %%
GPU = tch.cuda.is_available()
Num_Threads = 1
os.environ["OMP_NUM_THREADS"] = str(Num_Threads)
os.environ["MKL_NUM_THREADS"] = str(Num_Threads)
tch.set_num_threads(Num_Threads)
tch.set_num_interop_threads(Num_Threads)
# %%
dir_utils_AML = "Pycs_Win_3.11/"
dir_progs = "Pycs_Win_3.11/"
sys.path.append(os.path.normpath(dir_utils_AML))
sys.path.append(os.path.normpath(dir_progs))


from Utils_AML_Oct25 import  Metrics_f, SIFold, XR_f
from MMMLP_v8_v6 import IKFCV_f, MMMLP

# %%
dir = os.getcwd()
dir_datasets = os.path.normpath(os.path.join(dir, "../data-PP/"))
datasets = ["data-PP-variety", "data-PP-RIL"]
datanumber = 0
dir_dataset = os.path.join(dir_datasets, datasets[datanumber] + ".RData")
OD = pyreadr.read_r(dir_dataset)

K = 10
markers = OD["markers"]
pheno = OD["pheno"]
nirs = OD["nirs"]
drone = OD["df_smm_drones"]

for df in [pheno, nirs, drone]:
    df.index = df["year"].astype(int).astype(str) + "_" + df["genotype"]

drone = drone.loc[pheno.index]
inputs = {"markers": markers.copy(), "nirs": nirs.copy(), "drone": drone.copy(),
          "weather": OD["weather_smms"].copy()}
inputs["weather"].index = inputs["weather"]["year"]

scalers = {mod: StandardScaler() for mod in inputs}
inputs["markers"] = pd.DataFrame(scalers["markers"].fit_transform(inputs["markers"]),
                                 index=inputs["markers"].index,
                                 columns=inputs["markers"].columns)
for m in ["nirs", "drone"]:
    inputs[m].iloc[:, 3:] = scalers[m].fit_transform(inputs[m].iloc[:, 3:])
# %%

# Example
trait = "gpc"
predictor = "genotype"

y = pheno[trait]
pos_notna = y.notna()
y = y.loc[pos_notna]
pheno_r = pheno.loc[pos_notna].copy()
nirs_r = inputs["nirs"].loc[pos_notna]
drone_r = inputs["drone"].loc[pos_notna]
pheno_r["year"] = pheno_r["year"].astype(float)

pos_years = inputs["weather"].index.get_indexer(pheno_r["year"])
weather_r = inputs["weather"].iloc[pos_years]
activation_function = tch.nn.ReLU()
device = tch.device("cuda" if GPU else "cpu")
hypers_fixed = {"Epochs": 5, "Batch_size": None, "Seed": 42, "Iters": 5, "IK": 10,
                "Use": 10, "ResNet": True, "AF": activation_function, "Device": device}

dir_outs0 = os.path.join(dir, "dataset_" + datasets[datanumber].replace("-PP", ""),
                         "Model_v8_v6", "trait_" + trait + "-CV1")
os.makedirs(dir_outs0, exist_ok=True)

x_markers = inputs["markers"].loc[pheno_r["genotype"]]
x_nirs = nirs_r.iloc[:, 3:]
x_drone = drone_r.iloc[:, 3:]
x_weather = weather_r.iloc[:, 1:]
pred_dict = {"genotype": XR_f(x_markers.to_numpy()), "NIRs": XR_f(x_nirs.to_numpy()),
             "Drone": x_drone.to_numpy(), "Weather": x_weather.to_numpy()}

dir_folds = os.path.join(dir_outs0, "../../folds_df_dataset_" + datasets[datanumber] + "-CV1.csv")
folds_df = pd.read_csv(dir_folds)
folds = folds_df.loc[folds_df["trait"] == trait, "fold"].to_numpy()
predictor_names = predictor.split("+")
gids = pheno_r["genotype"].unique()


def save_csv(df, dir_outs):
    df.to_csv(dir_outs, mode="a", header=not os.path.exists(dir_outs), index=False)


for ok in range(1, K + 1):
    pos_tr = pheno_r["genotype"].isin(gids[folds != ok])
    pos_tst = pheno_r["genotype"].isin(gids[folds == ok])
    x_ls_tr = [pred_dict[pred][pos_tr, :] for pred in predictor_names]
    y_tr = y.loc[pos_tr].to_numpy().reshape(-1, 1)

    hypers_fixed["Batch_size"] = len(y_tr)
    dat_tr = {"x_ls": x_ls_tr, "y": y_tr}
    hypers_fixed["IFold"] = SIFold(dat_tr["y"], K=hypers_fixed["IK"], nq=10,
                                    random_state=hypers_fixed["Seed"])
    Time = time.time()
    No_Modalities = len(dat_tr["x_ls"])

    def f_O(**kwargs):
        nHLs = [kwargs["nHL" + str(j)] for j in range(No_Modalities)]
        Units = [kwargs["Units" + str(j)] for j in range(No_Modalities)]
        Ind = [Units[j] / (2 ** (nHLs[j] - 1)) > 8 for j in range(No_Modalities)]

        if No_Modalities > 1:
            nHLB2, UnitsB2 = kwargs["nHLB2"], kwargs["UnitsB2"]
            Ind = np.all(Ind) and UnitsB2 / (2 ** (nHLB2 - 1)) > 8
        else:
            Ind = np.all(Ind)
        if not Ind:
            return -10

        nHLs = [int(x) for x in nHLs]
        Units = [int(x) for x in Units]
        Units_ls = [[int(Units[m] / (2**j)) for j in range(nHL)]
                    for m, nHL in enumerate(nHLs)]
        UnitsB2_ls = ([int(UnitsB2 / (2**j)) for j in range(int(nHLB2))]
                      if No_Modalities > 1 else None)

        hypers_r = {"UnitsB1_lsls": Units_ls, "UnitsB2_ls": UnitsB2_ls,
                    "l": np.exp(kwargs["ll"]), "lr": np.exp(kwargs["llr"]),
                    "wd": np.exp(kwargs["lwd"])}
        hypers_p = {**hypers_fixed, **hypers_r}
        Val_metric = IKFCV_f(dat_tr, hypers_p, IK=hypers_fixed["IK"],
                             Use=hypers_fixed["Use"], PlotTraining=False)
        MSEP_Val = np.mean(Val_metric["Val_metric"])
        return MSEP_Val if np.isfinite(MSEP_Val) else -10

    Bounds = {"ll": (-10, 1), "llr": (-10, 0),
              "lwd": [np.log(0.05), np.log(0.95)]}
    for i in range(No_Modalities):
        Bounds[f"nHL{i}"], Bounds[f"Units{i}"] = [1, 4], [16, 512]
    if No_Modalities > 1:
        Bounds["nHLB2"], Bounds["UnitsB2"] = [1, 3], [16, 64]

    xi = 0.01
    acq = ExpectedImprovement(xi=xi)
    BO = BayesianOptimization(f=f_O, pbounds=Bounds, acquisition_function=acq,
                              random_state=5, verbose=1)
    BO.maximize(init_points=5, n_iter=hypers_fixed["Iters"])

    Pars_O = BO.max["params"]
    nHLs = [int(Pars_O["nHL" + str(j)]) for j in range(No_Modalities)]
    Units = [int(Pars_O["Units" + str(j)]) for j in range(No_Modalities)]
    Units_ls = [[int(Units[m] / (2**j)) for j in range(nHL)]
                for m, nHL in enumerate(nHLs)]
    UnitsB2_ls = ([int(Pars_O["UnitsB2"] / (2**j))
                   for j in range(int(Pars_O["nHLB2"]))]
                  if No_Modalities > 1 else None)

    hypers_r_o = {"UnitsB1_lsls": Units_ls, "UnitsB2_ls": UnitsB2_ls,
                  "l": np.exp(Pars_O["ll"]), "lr": np.exp(Pars_O["llr"]),
                  "wd": np.exp(Pars_O["lwd"])}
    params_o = {**hypers_fixed, **hypers_r_o}

    y_tst = y.loc[pos_tst]
    x_ls_tst = [tch.as_tensor(pred_dict[j][pos_tst, :], dtype=tch.float32,
                              device=params_o["Device"]) for j in predictor_names]
    yp_tst = np.zeros((len(y_tst), 1))

    for ik in range(1, hypers_fixed["Use"] + 1):
        pos_ival = hypers_fixed["IFold"] == ik
        dat_itr = {"x_ls": [arr[~pos_ival] for arr in dat_tr["x_ls"]],
                   "y": dat_tr["y"][~pos_ival]}
        DModel_O = MMMLP(params_o, dat_itr)
        DModel_O.fit(data_val=None, PlotTraining=False)
        DModel_O.Model.eval()
        with tch.no_grad():
            yp_tst += DModel_O.Model(x_ls_tst).cpu().numpy()

    yp_tst /= hypers_fixed["Use"]
    Time = time.time() - Time
    y_tst = y_tst.to_numpy().reshape(-1, 1)

    Id_df = {"Iters": hypers_fixed["Iters"], "Batch_size": hypers_fixed["Batch_size"],
             "GPU": str(GPU), "ResNet": str(hypers_fixed["ResNet"]),
             "AF": str(hypers_fixed["AF"]), "UF": "EI", "Xi": xi, "Folds": K,
             "IKUsed": str(hypers_fixed["IK"]) + "Used" + str(hypers_fixed["Use"]),
             "DataSet": datasets[datanumber], "Trait": trait}
    Tab_ok = pd.DataFrame({**Id_df, "Year": "NA", "Fold": ok, "Env": "Across all envs",
                           "Cor": np.corrcoef(y_tst[:, 0], yp_tst[:, 0])[0, 1],
                           "MSE": np.mean((y_tst - yp_tst) ** 2),
                           "NRMSE": np.sqrt(np.mean((y_tst - yp_tst) ** 2)) / np.mean(y_tst),
                           "Time": Time / 3600}, index=[0])

    df_Preds = pd.DataFrame({"Env": pheno_r.loc[pos_tst, "year"], "Fold": ok,
                             "y": y_tst[:, 0], "yp": yp_tst[:, 0]})
    df_Preds.reset_index(inplace=True)
    Tab_ok2 = Metrics_f(df_Preds, Grp="Env")
    Tab_ok2.insert(0, "Fold", ok)
    Tab_ok = pd.concat([Tab_ok, Tab_ok2])

    dir_outs = os.path.join(dir_outs0, "Preds_" + predictor + "-Trait_" + trait + ".csv")
    save_csv(df_Preds, dir_outs)
    dir_outs = os.path.join(dir_outs0, "Tab-Smm_" + predictor + "-Trait_" + trait + ".csv")
    save_csv(Tab_ok, dir_outs)

    Hyper_df_ok = pd.DataFrame(BO.max)
    Hyper_df_ok["Parameter"], Hyper_df_ok["Fold"] = Hyper_df_ok.index, ok
    dir_outs = os.path.join(dir_outs0, "Hypers_" + predictor + "-Trait_" + trait + ".csv")
    save_csv(Hyper_df_ok, dir_outs)
    print("Fold", ok)

# %%
