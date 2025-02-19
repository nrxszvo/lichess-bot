import os
import pathlib

import chess
import numpy as np
import torch

from .config import get_config
from .dual_zero.model import ModelArgs, Transformer
from .lib import STARTMV, BoardState, IllegalMoveException


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inp):
        return self.model(inp)


def get_model_args(cfgyml):
    model_args = ModelArgs(cfgyml.model_args)
    if cfgyml.elo_params["predict"]:
        model_args.gaussian_elo = cfgyml.elo_params["loss"] == "gaussian_nll"
        if cfgyml.elo_params["loss"] == "cross_entropy":
            model_args.elo_pred_size = len(cfgyml.elo_params["edges"]) + 1
        elif cfgyml.elo_params["loss"] == "gaussian_nll":
            model_args.elo_pred_size = 2
        elif cfgyml.elo_params["loss"] == "mse":
            model_args.elo_pred_size = 1
        else:
            raise Exception("did not recognize loss function name")
    model_args.n_timecontrol_heads = len(
        [n for _, grp in cfgyml.tc_groups.items() for n in grp]
    )
    return model_args


class MimicTestBotCore:
    def __init__(self, logger):
        dn = pathlib.Path(__file__).parent.resolve()
        cfg = os.path.join(dn, "dual_zero", "cfg.yml")
        cfgyml = get_config(cfg)
        self.tc_groups = cfgyml.tc_groups
        self.whiten_params = cfgyml.whiten_params
        model_args = get_model_args(cfgyml)
        self.model = Wrapper(Transformer(model_args))
        cp = torch.load(
            os.path.join(dn, "dual_zero", "weights.ckpt"),
            map_location=torch.device("cpu"),
        )
        self.model.load_state_dict(cp)
        self.model.eval()
        self.board = BoardState()
        self.inp = torch.tensor([[STARTMV]], dtype=torch.int32)
        self.ms = []
        self.logger = logger

    def _add_move(self, mvid):
        mv = torch.tensor([[mvid]], dtype=torch.int32)
        self.inp = torch.cat([self.inp, mv], dim=1)

    def _print_top_five(self, pred_grps):
        tcid = 0
        self.logger.info("...Top Five Moves Per TC Group...")
        for tcg, incgs in sorted(self.tc_groups.items()):
            for inc in sorted(incgs):
                self.logger.info(f"{tcg}.{inc}:")
                probs, mvids = pred_grps[tcid].softmax(dim=0).sort(descending=True)
                for i, (p, mvid) in enumerate(zip(probs, mvids)):
                    if i == 5:
                        break
                    try:
                        uci = self.board.mvid_to_uci(mvid)
                        self.logger.info(f"\t{i}. {uci} ({100 * p.item():.2f})")
                    except IllegalMoveException:
                        pass

    @torch.inference_mode
    def predict(self, uci: str) -> chess.Move:
        wm, ws = self.whiten_params

        def parse_elo(elo_pred):
            m = elo_pred[0, :, 0, 0] * ws + wm
            s = ((elo_pred[0, :, 0, 1] ** 0.5) * ws) ** 2
            return m, s

        if uci is not None:
            mvid = self.board.uci_to_mvid(uci)
            self.board.update(mvid)
            self.logger.info(self.board.print())
            self._add_move(mvid)

        mv_pred, elo_pred = self.model(self.inp)

        def_elo = {"m": wm, "s": ws**2}
        info = {"weloParams": def_elo, "beloParams": def_elo}
        if uci is not None:
            ms, ss = parse_elo(elo_pred)
            if len(ms) % 2 == 0:
                widx = -2
                bidx = -1
            else:
                widx = -1
                bidx = -2
            info = {
                "weloParams": {"m": ms[widx].item(), "s": ss[widx].item()},
                "beloParams": {"m": ms[bidx].item(), "s": ss[bidx].item()},
            }
            self.logger.info(f"White Elo: {ms[widx]} +/- {ss[widx]}")
            self.logger.info(f"Black Elo: {ms[bidx]} +/- {ss[bidx]}")
        # self._print_top_five(mv_pred[0, -1])
        probs, mvids = mv_pred[0, -1, -1].softmax(dim=0).sort(descending=True)
        p = probs[:5].double() / probs[:5].double().sum()
        p[p < 0.05] = 1e-8
        p = p / p.sum()
        mvids = np.random.choice(mvids[:5], size=5, replace=False, p=p)
        for mvid in mvids:
            try:
                mv = self.board.update(mvid)
                self._add_move(mvid)
                break
            except IllegalMoveException:
                continue
        else:
            raise Exception("model did not produce a legal move")
        return mv, info
