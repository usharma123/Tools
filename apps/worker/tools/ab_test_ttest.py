from typing import Any, Dict, Optional
import math, time
import numpy as np
from pydantic import BaseModel, model_validator
from fastapi import APIRouter
from scipy import stats

router = APIRouter()


class ABTTArgs(BaseModel):
    # Binary mode
    successes_a: Optional[int] = None
    trials_a: Optional[int] = None
    successes_b: Optional[int] = None
    trials_b: Optional[int] = None
    # Continuous mode
    mean_a: Optional[float] = None
    sd_a: Optional[float] = None
    n_a: Optional[int] = None
    mean_b: Optional[float] = None
    sd_b: Optional[float] = None
    n_b: Optional[int] = None

    alpha: float = 0.05
    two_tailed: bool = True
    equal_var: bool = False
    assume_independent: bool = True

    @model_validator(mode="after")
    def exactly_one_mode(self):
        binary = all(v is not None for v in [self.successes_a, self.trials_a, self.successes_b, self.trials_b])
        cont = all(v is not None for v in [self.mean_a, self.sd_a, self.n_a, self.mean_b, self.sd_b, self.n_b])
        assert binary ^ cont, "Provide either binary (successes/trials) OR continuous (mean/sd/n), not both."
        return self


@router.post("/tools/ab_test_ttest")
def ab_test_ttest(args: ABTTArgs) -> Dict[str, Any]:
    t0 = time.perf_counter()
    out: Dict[str, Any] = {"tool_version": "ab_test_ttest/0.2.0", "alpha": args.alpha, "two_tailed": args.two_tailed}

    if all(v is not None for v in [args.successes_a, args.trials_a, args.successes_b, args.trials_b]):
        sa, na = int(args.successes_a), int(args.trials_a)
        sb, nb = int(args.successes_b), int(args.trials_b)
        assert 0 <= sa <= na and 0 <= sb <= nb and na > 0 and nb > 0, "Invalid successes/trials."
        pa, pb = sa / na, sb / nb
        diff = pb - pa
        tails = 2 if args.two_tailed else 1
        p_pool = (sa + sb) / (na + nb)
        se_H0 = math.sqrt(p_pool * (1 - p_pool) * (1 / na + 1 / nb))
        z = diff / se_H0 if se_H0 > 0 else float("inf")
        p_value = (1 - stats.norm.cdf(abs(z))) * tails
        zcrit = stats.norm.ppf(1 - args.alpha / tails)
        se_diff = math.sqrt(pa * (1 - pa) / na + pb * (1 - pb) / nb)
        ci = [diff - zcrit * se_diff, diff + zcrit * se_diff]
        se_a = math.sqrt(pa * (1 - pa) / na)
        se_b = math.sqrt(pb * (1 - pb) / nb)
        ci_a = [max(0.0, pa - zcrit * se_a), min(1.0, pa + zcrit * se_a)]
        ci_b = [max(0.0, pb - zcrit * se_b), min(1.0, pb + zcrit * se_b)]
        out.update({
            "mode": "binary",
            "group_a": {"successes": sa, "trials": na, "rate": pa, "ci": ci_a},
            "group_b": {"successes": sb, "trials": nb, "rate": pb, "ci": ci_b},
            "effect": {"name": "absolute_diff", "value": diff, "ci": ci},
            "relative_lift": (diff / pa) if pa > 0 else None,
            "test_stat": {"z": z},
            "p_value": p_value,
            "assumptions": {"independent_samples": args.assume_independent, "large_sample_normal": True},
        })
    else:
        ma, sda, na = float(args.mean_a), float(args.sd_a), int(args.n_a)
        mb, sdb, nb = float(args.mean_b), float(args.sd_b), int(args.n_b)
        assert na > 1 and nb > 1 and sda >= 0 and sdb >= 0, "Invalid mean/sd/n."
        diff = mb - ma
        tails = 2 if args.two_tailed else 1
        if args.equal_var:
            sp2 = ((na - 1) * sda * sda + (nb - 1) * sdb * sdb) / (na + nb - 2)
            se = math.sqrt(sp2 * (1 / na + 1 / nb))
            df = na + nb - 2
        else:
            se2 = (sda * sda) / na + (sdb * sdb) / nb
            se = math.sqrt(se2)
            df = (se2 * se2) / (((sda * sda) / (na * na * (na - 1))) + ((sdb * sdb) / (nb * nb * (nb - 1))))
        t = diff / se if se > 0 else float("inf")
        p_value = (1 - stats.t.cdf(abs(t), df)) * tails
        tcrit = stats.t.ppf(1 - args.alpha / tails, df)
        ci = [diff - tcrit * se, diff + tcrit * se]
        ci_a = [ma - tcrit * (sda / math.sqrt(na)), ma + tcrit * (sda / math.sqrt(na))]
        ci_b = [mb - tcrit * (sdb / math.sqrt(nb)), mb + tcrit * (sdb / math.sqrt(nb))]
        out.update({
            "mode": "continuous",
            "group_a": {"mean": ma, "sd": sda, "n": na, "ci": ci_a},
            "group_b": {"mean": mb, "sd": sdb, "n": nb, "ci": ci_b},
            "effect": {"name": "mean_diff", "value": diff, "ci": ci},
            "test_stat": {"t": t, "df": df},
            "p_value": p_value,
            "assumptions": {"independent_samples": args.assume_independent, "welch": not args.equal_var, "normal_or_clt": True},
        })

    out["runtime_ms"] = (time.perf_counter() - t0) * 1000.0
    return out


