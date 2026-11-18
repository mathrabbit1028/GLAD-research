"""Fixed-ratio mixture objectives; all optimizers operate on small CPU arrays."""
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.special import logsumexp


def project_capped_simplex(target, cap):
    """Euclidean projection: clip(target - multiplier, 0, cap), sum = 1."""
    target = np.asarray(target, dtype=float)
    if target.ndim != 1 or not np.isfinite(target).all() or cap <= 0 or len(target) * cap < 1 - 1e-12:
        raise ValueError('Invalid capped simplex projection')
    shifted = target - target.max()
    lower, upper = float(shifted.min() - cap), 0.
    for _ in range(100):
        midpoint = (lower + upper) / 2
        if np.clip(shifted - midpoint, 0, cap).sum() > 1:
            lower = midpoint
        else:
            upper = midpoint
    result = np.clip(shifted - (lower + upper) / 2, 0, cap)
    if abs(result.sum() - 1) > 1e-8:
        raise RuntimeError('Capped simplex projection lost numerical precision')
    return result


def objective(log_ratios, weights):
    logs = np.full_like(weights, -np.inf, dtype=float)
    np.log(weights, out=logs, where=weights > 0)
    return float(logsumexp(log_ratios + logs, axis=1).mean())


def gradient(log_ratios, weights):
    logs = np.full_like(weights, -np.inf, dtype=float)
    np.log(weights, out=logs, where=weights > 0)
    denominator = logsumexp(log_ratios + logs, axis=1)
    return np.exp(log_ratios - denominator[:, None]).mean(0)


def normal_weights(log_ratios, steps, tolerance):
    weights = np.full(log_ratios.shape[1], 1 / log_ratios.shape[1])
    previous = objective(log_ratios, weights)
    for step in range(steps):
        logs = np.full_like(weights, -np.inf)
        np.log(weights, out=logs, where=weights > 0)
        posterior = log_ratios + logs
        weights = np.exp(posterior - logsumexp(posterior, axis=1, keepdims=True)).mean(0)
        weights /= weights.sum()
        current = objective(log_ratios, weights)
        if abs(current - previous) < tolerance:
            break
        previous = current
    return weights, dict(objective=current, steps=step + 1)


def abnormal_weights(log_ratios, shell_rates, settings, rng, progress=None):
    k = log_ratios.shape[1]
    vertex = int(np.argmin(log_ratios.mean(0)))
    literal = np.eye(k)[vertex]
    if settings.abnormal_mode == 'literal':
        return literal, dict(literal_component=vertex, objective=objective(log_ratios, literal))
    a = np.asarray(shell_rates)
    cap, minimum = settings.cap, settings.min_shell_rate
    feasibility = linprog(-a, A_eq=np.ones((1, k)), b_eq=[1.], bounds=[(0., cap)] * k, method='highs')
    if not feasibility.success or a @ feasibility.x < minimum - 1e-8:
        maximum = float(a @ feasibility.x) if feasibility.success else None
        raise ValueError(f'Abnormal mixture infeasible: maximum capped shell rate={maximum}, required={minimum}')
    constraints = [dict(type='eq', fun=lambda w: w.sum() - 1, jac=lambda w: np.ones(k)),
                   dict(type='ineq', fun=lambda w: a @ w - minimum, jac=lambda w: a)]

    def project(target, initial):
        capped = project_capped_simplex(target, cap)
        if a @ capped >= minimum - 1e-10:
            return capped
        # Remove the irrelevant common offset before the general QP solve.
        target = target - target.mean()
        result = minimize(lambda w: 0.5 * np.square(w - target).sum(), initial,
                          jac=lambda w: w - target, method='SLSQP', bounds=[(0., cap)] * k,
                          constraints=constraints, options=dict(ftol=1e-11, maxiter=200))
        if (not result.success or abs(result.x.sum() - 1) > 1e-7 or
                a @ result.x < minimum - 1e-7):
            raise RuntimeError(f'Weight projection failed: {result.message}')
        return result.x

    best, best_value, values = None, float('inf'), []
    for restart in range(settings.restarts):
        weights = project(np.full(k, 1 / k) if restart == 0 else rng.dirichlet(np.ones(k)), feasibility.x)
        value = objective(log_ratios, weights)
        for _ in range(settings.abnormal_steps):
            grad = gradient(log_ratios, weights)
            rate, accepted = 1. / max(1., np.max(np.abs(grad))), False
            for _ in range(20):
                candidate = project(weights - rate * grad, weights)
                proposed = objective(log_ratios, candidate)
                if proposed <= value + 1e-12:
                    accepted = True
                    break
                rate *= 0.5
            if not accepted:
                break
            change = abs(proposed - value)
            weights, value = candidate, proposed
            if change < settings.tolerance:
                break
        values.append(value)
        if value < best_value:
            best, best_value = weights.copy(), value
        if progress:
            progress(restart + 1, value)
    return best, dict(objective=best_value, literal_component=vertex,
                     literal_objective=objective(log_ratios, literal), restart_objectives=values,
                     shell_rate=float(a @ best), max_weight=float(best.max()),
                     simplex_residual=float(abs(best.sum() - 1)))
