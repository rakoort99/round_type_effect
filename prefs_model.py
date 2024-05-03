import torch
from tqdm import tqdm
import pyro
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
from pyro import poutine
from pyro.optim import ClippedAdam, SGD, NAdam
from pyro.infer import (
    SVI,
    TraceGraph_ELBO,
)
from collections import defaultdict
import numpy as np
import torchsort
import argparse

parser = argparse.ArgumentParser("prefs_model")
parser.add_argument(
    "records",
    help="relevant records, as defined in gen_model_inputs. options are 'full', 'pre', 'op_pre', or 'toy'",
    type=str,
    nargs='?',
    const="op_pre",
    default="op_pre",
)
parser.add_argument(
    "pool_type",
    help="method of calculating pool. 'rw' for round-wise calculation, 'tw' for tournament-wise calculation",
    type=str,
    nargs='?',
    const="rw",
    default="rw",
)
parser.add_argument(
    "beta_warmup",
    help="epoch to begin learning multinomial regression coefficients",
    type=int,
    nargs='?',
    const=8000,
    default=8000,
)
parser.add_argument(
    "batch_size",
    help="batch size for learning",
    type=int,
    nargs='?',
    const=20000,
    default=20000,
)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
pyro.enable_validation(True)
eps = 10**-3.19


# just loads data into a big tuple
def load_model_data(records="full", pools="rw"):
    prefix = ""
    pools_prefix = ""
    if records == "full":
        pass
    else:
        prefix = records + "_"
        pools_prefix = records + "_"
    if pools == "rw":
        pools_prefix = pools_prefix + "rw_"
    with torch.no_grad():
        affs = torch.load(f"data/test/{prefix}affs.pkl").to(device)
        negs = torch.load(f"data/test/{prefix}negs.pkl").to(device)
        judges = torch.load(f"data/test/{prefix}judges.pkl").to(device)
        pools = torch.load(f"data/test/{pools_prefix}pools.pkl").to(device)
        judgelist = (
            torch.from_numpy(torch.load(f"data/test/{prefix}judge_list.pkl"))
            .type(torch.FloatTensor)
            .to(device)
        )
        teamlist = (
            torch.from_numpy(torch.load(f"data/test/{prefix}team_list.pkl"))
            .type(torch.FloatTensor)
            .to(device)
        )
    return (affs, negs, judges, pools, teamlist, judgelist)


data_tuple = load_model_data(args.records, args.pool_type)
data_size = len(data_tuple[0])


class prefs_model_lin:
    def __init__(self, data_tuple, bsize=None, debug=False, betastart=None):
        self.affs, self.negs, self.judges, self.pools, self.teamlist, self.judgelist = (
            data_tuple
        )
        self.epoch = 0
        self.npart = len(self.teamlist)
        self.njudge = len(self.judgelist)
        self.nround = len(self.affs)
        self.beta_awake = False if betastart else True
        self.judge_awake = True
        self.team_awake = True
        self.beta_start = betastart
        self.bsize = bsize if bsize else self.nround
        self.debug = debug
        self.d = lambda x, y: (x.unsqueeze(-1) - y.unsqueeze(0)) ** 2
        self.loss_list = []
        self.gradient_norms = defaultdict(list)
        self.min_loss = 1e6

    @poutine.scale(scale=1.0 / data_size)  # scale by number of observed rounds
    def model(self, annealing_factor=1.0):
        if self.debug:
            print("In the model")

        ## define multinomial regression coeffs
        beta_pref = pyro.param(
            "beta_pref", torch.tensor(-4.0), constraint=constraints.less_than(-1.0)
        )
        beta_mut = pyro.param(
            "beta_mut", torch.tensor(-3.0), constraint=constraints.less_than(-0.5)
        )
        if not self.beta_awake:
            beta_pref = beta_pref.detach()
            beta_mut = beta_mut.detach()

        if self.debug:
            print(beta_pref.shape)
            print("passed regression params")

        ## Construct plates
        plate_pairings = pyro.plate(
            "pairings", self.nround, dim=-1, subsample_size=self.bsize
        )
        plate_judges = pyro.plate("judges", self.njudge)
        plate_teams = pyro.plate("teams", self.npart)

        ## sample team ideology mixture weights.
        with plate_teams:
            team_weights = pyro.sample(
                "team_weights", dist.Beta(torch.tensor(1.0), torch.tensor(1.0))
            )
        if self.debug:
            print("team_weights:")
            print("shape:", team_weights.shape)

        ## sample judge ideology mixture weights.
        with plate_judges:
            judge_weights = pyro.sample(
                "judge_weights", dist.Beta(torch.tensor(1.0), torch.tensor(1.0))
            )
        if self.debug:
            print("jscores:")
            print("shape:", judge_weights.shape)

        with plate_pairings as ind:
            ## get batch of teams and judge pools
            ind = ind.to(device)
            aff = self.affs.index_select(0, ind)
            neg = self.negs.index_select(0, ind)
            pool = self.pools.index_select(0, ind).type(torch.float)
            if self.debug:
                print("pool shape", pool.shape)
            if self.debug:
                print("aff shape:", aff.shape)

            aff_weights = team_weights.index_select(-1, aff).squeeze()
            neg_weights = team_weights.index_select(-1, neg).squeeze()

            if self.debug:
                print("aff_weights shape:", aff_weights.shape)

            ## get ideological distance
            aff_prefs = self.d(aff_weights, judge_weights)
            neg_prefs = self.d(neg_weights, judge_weights)
            if self.debug:
                print("aff pref shape", aff_prefs.shape)

            ## get preferences, quantile-ized for given pool
            aff_avail = torchsort.soft_rank(
                aff_prefs + 10 * (1 - pool), regularization_strength=eps
            ) / pool.sum(dim=-1, keepdim=True)
            neg_avail = torchsort.soft_rank(
                neg_prefs + 10 * (1 - pool), regularization_strength=eps
            ) / pool.sum(dim=-1, keepdim=True)
            if self.debug:
                print("avail shape", aff_avail.shape)

            ## get mutuality, preferredness penalties
            mutuality = torch.abs(aff_avail - neg_avail)
            preferredness = aff_avail + neg_avail

            ## get categorical probabilities
            cat_p = beta_pref * preferredness + beta_mut * mutuality + -1e6 * (1 - pool)
            if self.debug:
                print("weights shape", cat_p.shape)

            ## sample
            obs = pyro.sample(
                "obs",
                dist.Categorical(logits=cat_p, validate_args=False),
                obs=self.judges.index_select(-1, ind),
            )
            if self.debug:
                print(obs.shape)

    @poutine.scale(scale=1 / data_size)
    def guide(self, annealing_factor=1.0):
        if self.debug:
            print("In the guide")
        self.epoch += 1

        ## Construct plates
        plate_pairings = pyro.plate(
            "pairings", self.nround, dim=-1, subsample_size=self.bsize
        )
        plate_judges = pyro.plate("judges", self.njudge)
        plate_teams = pyro.plate("teams", self.npart)

        ## wake up betas if necessary
        if self.epoch == self.beta_start:
            self.beta_awake = True
            print("betas awake for the first time", self.epoch)

        ## get params for team ideal points
        team_a = pyro.param(
            "team_a",
            1.01 * torch.ones((self.npart,)),
            constraint=constraints.interval(1.0, 100.0),
        )
        team_b = pyro.param(
            "team_b",
            1.01 * torch.ones((self.npart,)),
            constraint=constraints.interval(1.0, 100.0),
        )
        if self.debug:
            print("team_concentrations:", team_a.shape)
        if not self.team_awake:
            team_a = team_a.detach()
            team_b = team_b.detach()

        ## sample team ideal points
        with plate_teams:
            with poutine.scale(scale=annealing_factor):
                team_weights = pyro.sample(
                    "team_weights",
                    dist.Beta(team_a, team_b),
                )
                # infer=dict(baseline={'use_decaying_avg_baseline': True,
                # 'baseline_beta': 0.9}))
            if self.debug:
                print("team weights:", team_weights.shape)

        ## get params for judge ideal points
        judge_a = pyro.param(
            "judge_a",
            1.01 * torch.ones((self.njudge,)),
            constraint=constraints.interval(1.0, 100.0),
        )
        judge_b = pyro.param(
            "judge_b",
            1.01 * torch.ones((self.njudge,)),
            constraint=constraints.interval(1.0, 100.0),
        )
        if not self.judge_awake:
            judge_a = judge_a.detach()
            judge_b = judge_b.detach()

        ## sample judge ideal points
        with plate_judges:
            if self.debug:
                print("judge_concentrations:", judge_a.shape)
            with poutine.scale(scale=annealing_factor):
                judge_weights = pyro.sample(
                    "judge_weights",
                    dist.Beta(judge_a, judge_b),
                )
                # infer=dict(baseline={'use_decaying_avg_baseline': True,
                # 'baseline_beta': 0.9}))
            if self.debug:
                print("judge weights:", judge_weights.shape)

        with plate_pairings as ind:  # noqa: F841
            pass

    def inference(
        self,
        max_step=5000,
        start_rate=0.01,
        opt="NAdam",
        num_part=1,
        early_stop=None,
        freeze=None,
    ):
        if not early_stop:
            early_stop = 1e7 / self.bsize

        ## freezes params we want to fix
        if freeze:
            if "betas" in freeze:
                self.beta_awake = False
            if "team_concentrations" in freeze:
                self.team_awake = False
            if "judge_concentrations" in freeze:
                self.judge_awake = False

        ## function sets optimizer-specific params
        ## also sets param-specific learning rates
        def per_param_callable(param_name):
            param_dict = {}
            if opt == "Nesterov":
                param_dict = param_dict | {"momentum": 0.9, "nesterov": True}
            if opt == "NAdam":
                param_dict = param_dict | {
                    "momentum_decay": 0.002,
                    "betas": (0.9, 0.99),
                }
            if param_name in ["beta_mut", "beta_pref"]:
                param_dict = param_dict | {"lr": start_rate * 0.3}
                return param_dict
            else:
                param_dict = param_dict | {"lr": start_rate}
                return param_dict

        if opt == "ClippedAdam":
            optimizer = ClippedAdam(per_param_callable)
        if opt == "Nesterov":
            optimizer = SGD(per_param_callable)
        if opt == "SGD":
            optimizer = SGD(per_param_callable)
        if opt == "NAdam":
            optimizer = NAdam(per_param_callable)

        ## define SVI params
        svi = SVI(
            self.model,
            self.guide,
            optimizer,
            loss=TraceGraph_ELBO(num_particles=num_part, vectorize_particles=False),
        )

        ## loop and learn
        print("Doing inference")
        ctr = 0
        KLA = 1.0
        for k in tqdm(range(1, max_step + 1)):
            ## take step
            # if k >= KL_start and KLA < 1.:
            #     KLA += KL_rate
            loss_k = svi.step(KLA)
            self.loss_list.append(loss_k)
            if self.epoch > self.beta_start:
                ctr += 1
            if k % 500 == 0:
                print("mean loss over last 500 steps:", np.mean(self.loss_list[-500:]))

            ## update best loss and reset exit counter
            if loss_k < self.min_loss:
                self.min_loss = loss_k
                ctr = 0

            ## exit condition:
            if ctr > early_stop:
                print(early_stop, "iterations with no improvement, exiting loop")
                print("terminal loss:", loss_k)
                return loss_k

            ## first run setup
            if k == 1:
                print("initial loss:", loss_k)

        print("terminal loss:", loss_k)
        return loss_k


def main():
    """
    trains model
    """
    model_name = (
        args.records
        + "_"
        + args.pool_type
        + "_"
        + f"_warmup{args.beta_warmup}"
        + f"_batch{args.batch_size}"
    )

    pm = prefs_model_lin(
        data_tuple, bsize=args.batch_size, debug=False, betastart=args.beta_warmup
    )
    losses = pm.inference(max_step=10000, start_rate=0.2, early_stop=3000)
    pyro.get_param_store().save("models/" + model_name + ".torch")

    losses = pm.inference(max_step=10000, start_rate=0.05, early_stop=3000)
    pyro.get_param_store().save("models/" + model_name + ".torch")

    losses = pm.inference(max_step=10000, start_rate=0.005, early_stop=3000)  # noqa: F841
    pyro.get_param_store().save("models/" + model_name + ".torch")


if __name__ == "__main__":
    main()
